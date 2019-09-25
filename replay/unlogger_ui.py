#!/usr/bin/env python
import argparse
import os
import sys
import pygame
import numpy as np
import cv2
import math
import time
import re
import zmq
import json
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket

from common.basedir import BASEDIR
os.environ['BASEDIR'] = BASEDIR
from cereal import log as capnp_log

from zmq_utils import send_array, recv_array

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route, CAMERA_FILENAMES, LOG_FILENAMES
from tools.lib.route_framereader import RouteFrameReader


from collections import namedtuple
from common.transformations.camera import eon_intrinsics, FULL_FRAME_SIZE
from common.transformations.model import MODEL_CX, MODEL_CY, MODEL_INPUT_SIZE
from selfdrive.config import UIParams as UP
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.latcontrol_helpers import compute_path_pinv, model_polyfit
from selfdrive.config import RADAR_TO_CENTER
from tools.lib.lazy_property import lazy_property
from tools.replay.lib.ui_helpers import to_lid_pt, draw_path, draw_steer_path, draw_mpc, \
                                                  draw_lead_car, draw_lead_on, init_plots, warp_points, find_color
from selfdrive.car.toyota.interface import CarInterface as ToyotaInterface
from common.transformations.model import get_camera_frame_from_model_frame

from osm_helpers import (
  load_results_from_file, lat_lon_to_screen_x_y, query_nodes, init_overpass_api, find_closest_nodes, parse_crossing,
  parse_traffic_sign, parse_traffic_signal, lat_lon_distance, mkdirs_exists_ok)

if False:
  sys.path.append('/home/nanami/servers/keras_test')
  from white_line_gt import find_white_lines

  sys.path.append('/home/nanami/ssd')
  from lisa_to_voc import dict_to_xml


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

_PATH_X = np.arange(101.)
_PATH_XD = np.arange(101.)
_PATH_PINV = compute_path_pinv(50)

ModelUIData = namedtuple("ModelUIData", ["cpath", "lpath", "rpath", "lead", "lead_std", "freepath"])

MAX_DISTANCE = 200.
MAP_VIEW_WIDTH_HEIGHT = 320
MAP_VIEW_LEFT = 52
MAP_VIEW_TOP = 330

class CalibrationTransformsForWarpMatrix(object):
  def __init__(self, model_to_full_frame, K, E):
    self._model_to_full_frame = model_to_full_frame
    self._K = K
    self._E = E

  @lazy_property
  def model_to_full_frame(self):
    return self._model_to_full_frame

  @lazy_property
  def car_to_model(self):
    return np.linalg.inv(self._model_to_full_frame).dot(self._K).dot(
      self._E[:, [0, 1, 3]])

  @lazy_property
  def car_to_screen(self):
    return self._K.dot(self._E[:, [0, 1, 3]])

  @lazy_property
  def car_to_screen_3d(self):
    return self._K.dot(self._E)

  # project car frame 3d pt to screen
  def car_project_to_screen(self, pt):
    screen_pt = self.car_to_screen_3d.dot(pt)
    return screen_pt[0] / screen_pt[2], screen_pt[1] / screen_pt[2]

  @lazy_property
  def car_to_bb(self):
    return self._K.dot(self._E[:, [0, 1, 3]])


def pygame_modules_have_loaded():
  return pygame.display.get_init() and pygame.font.get_init()

def draw_var(y, x, var, color, img, calibration, top_down):
  # otherwise drawing gets stupid
  var = max(1e-1, min(var, 0.7))

  varcolor = tuple(np.array(color)*0.5)
  draw_path(y - var, x, varcolor, img, calibration, top_down)
  draw_path(y + var, x, varcolor, img, calibration, top_down)


class ModelPoly(object):
  def __init__(self, model_path):
    if len(model_path.points) == 0:
      self.valid = False
      return

    self.poly = model_polyfit(model_path.points, _PATH_PINV)
    self.prob = model_path.prob
    self.std = model_path.std
    self.y = np.polyval(self.poly, _PATH_XD)
    self.valid = True

def extract_model_data(md):
  return ModelUIData(
    cpath=ModelPoly(md.model.path),
    lpath=ModelPoly(md.model.leftLane),
    rpath=ModelPoly(md.model.rightLane),
    lead=md.model.lead.dist,
    lead_std=md.model.lead.std,
    freepath=md.model.freePath)

def plot_model(m, VM, v_ego, curvature, imgw, calibration, top_down, d_poly, top_down_color=216):
  # Draw bar representing position and distribution of lead car from unfiltered vision model
  if top_down is not None:
    _, _ = to_lid_pt(m.lead, 0)
    _, py_top = to_lid_pt(m.lead + m.lead_std, 0)
    px, py_bottom = to_lid_pt(m.lead - m.lead_std, 0)
    top_down[1][int(round(px - 4)):int(round(px + 4)), py_top:py_bottom] = top_down_color

  if calibration is None:
    return

  if d_poly is not None:
    dpath_y = np.polyval(d_poly, _PATH_X)
    draw_path(dpath_y, _PATH_X, RED, imgw, calibration, top_down, RED)

  if m.lpath.valid:
    color = (0, int(255 * m.lpath.prob), 0)
    draw_path(m.lpath.y, _PATH_XD, color, imgw, calibration, top_down, YELLOW)
    draw_var(m.lpath.y, _PATH_XD, m.lpath.std, color, imgw, calibration, top_down)

  if m.rpath.valid:
    color = (0, int(255 * m.rpath.prob), 0)
    draw_path(m.rpath.y, _PATH_XD, color, imgw, calibration, top_down, YELLOW)
    draw_var(m.rpath.y, _PATH_XD, m.rpath.std, color, imgw, calibration, top_down)

  if len(m.freepath) > 0:
    for i, p in enumerate(m.freepath):
      d = i*2
      px, py = to_lid_pt(d, 0)
      cols = [36, 73, 109, 146, 182, 219, 255]
      if p >= 0.4:
        top_down[1][int(round(px - 4)):int(round(px + 4)), int(round(py - 4)):int(round(py + 4))] = find_color(top_down[0], (0, cols[int((p-0.4)*10)], 0))
      elif p <= 0.2:
        top_down[1][int(round(px - 4)):int(round(px + 4)), int(round(py - 4)):int(round(py + 4))] = 192 #find_color(top_down[0], (192, 0, 0))

  # draw user path from curvature
  draw_steer_path(v_ego, curvature, BLUE, imgw, calibration, top_down, VM, BLUE)


def maybe_update_radar_points(lt, lid_overlay):
  ar_pts = []
  if lt is not None:
    ar_pts = {}
    for track in lt.liveTracks:
      ar_pts[track.trackId] = [track.dRel, track.yRel, track.vRel, track.aRel, track.oncoming, track.stationary]
  for ids, pt in ar_pts.viewitems():
    px, py = to_lid_pt(pt[0], pt[1])
    if px != -1:
      if pt[-1]:
        color = 240
      elif pt[-2]:
        color = 230
      else:
        color = 255
      if int(ids) == 1:
        lid_overlay[px - 2:px + 2, py - 10:py + 10] = 100
      else:
        lid_overlay[px - 2:px + 2, py - 2:py + 2] = color

def get_blank_lid_overlay(UP):
  lid_overlay = np.zeros((UP.lidar_x, UP.lidar_y), 'uint8')
  # Draw the car.
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)):int(
    round(UP.lidar_car_x + UP.car_hwidth)), int(round(UP.lidar_car_y -
                                                      UP.car_front))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)):int(
    round(UP.lidar_car_x + UP.car_hwidth)), int(round(UP.lidar_car_y +
                                                      UP.car_back))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)), int(
    round(UP.lidar_car_y - UP.car_front)):int(round(
      UP.lidar_car_y + UP.car_back))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x + UP.car_hwidth)), int(
    round(UP.lidar_car_y - UP.car_front)):int(round(
      UP.lidar_car_y + UP.car_back))] = UP.car_color
  return lid_overlay


def draw_transparent_text(font, t, color):
  txt_surf = font.render(t, True, color)
  # Create a transparent surface.
  alpha_img = pygame.surface.Surface(txt_surf.get_size(), pygame.SRCALPHA)
  # Fill it with white and the desired alpha value.
  alpha_img.fill((255, 255, 255, 140))
  # Blit the alpha surface onto the text surface and pass BLEND_RGBA_MULT.
  txt_surf.blit(alpha_img, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
  return txt_surf

def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Unlogger and UI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("data_dir", nargs='?', default="/home/nanami/data/Chunk_2/b0c9d2329ad1606b|2018-10-09--14-06-32",
                        help="Path to route directory.")
  parser.add_argument("--lat", nargs="?", type=float, default=37.7216742,
                      help="Latitude")
  parser.add_argument("--lon", nargs="?", type=float, default=-122.4691069,
                      help="Longitude")
  parser.add_argument("--save_img", nargs="?", type=int, default=0,
                      help="Save images for training")
  parser.add_argument("--save_img_dir", nargs='?', default='/home/nanami/data',
                      help="Path to directory in which images are saved.")
  parser.add_argument("--multi_routes", nargs="?", type=int, default=0,
                      help="data_dir is the root for multiple routes")
  parser.add_argument("--whiteline", nargs="?", type=int, default=0,
                      help="Detect whitelines")
  parser.add_argument("--tl", nargs="?", type=int, default=0,
                      help="Detect traffic lights")
  parser.add_argument("--out_imgs_dir", nargs="?", default='/home/nanami/data/comma2k19/images',
                      help="Traffic lights images dir")
  parser.add_argument("--out_annos_dir", nargs="?", default='/home/nanami/data/comma2k19/annotations',
                      help="Traffic lights images dir")
  return parser


class SimpleEcho(WebSocket):

  def handleMessage(self):
    # echo message back to client
    self.sendMessage(self.data)

  def handleClose(self):
    print(self.address, 'closed')

def main(argv):
  args = get_arg_parser().parse_args(sys.argv[1:])
  if not args.data_dir:
    print('Data directory invalid.')
    return
  mkdirs_exists_ok(args.save_img_dir)
  mkdirs_exists_ok(args.out_imgs_dir)
  mkdirs_exists_ok(args.out_annos_dir)

  '''
  # comma2k19 data:
  # log dir: ~/data/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/10/
  # data_dir: ~/data/Chunk_1
  # route_name: b0c9d2329ad1606b|2018-07-27--06-03-57
  # Segment: b0c9d2329ad1606b|2018-07-27--06-03-57--10
  # 
  # Eon logs:
  # log dir: ~/data/eon_logs/855194471f3a9afd/2019-07-11--12-16-43--11
  # data_dir: ~/data/eon_logs
  # route_name: 855194471f3a9afd|2019-07-11--12-16-43
  # Segment: 855194471f3a9afd|2019-07-11--12-16-43--11
  '''
  if args.multi_routes:
    data_dir = args.data_dir
    route_names = [name for name in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, name))]
  else:
    if os.path.basename(args.data_dir) == 'eon_logs':
      data_dir = args.data_dir
      routes = set()
      for name in os.listdir(args.data_dir):
        route_name_prefix = name
        seg_name_re = r'([0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2})--[0-9]+'
        for seg_name in os.listdir(os.path.join(args.data_dir, name)):
          m = re.match(seg_name_re, seg_name)
          if m:
            log_video_files = os.listdir(os.path.join(args.data_dir, name, seg_name))
            if any([cf for cf in CAMERA_FILENAMES if cf in log_video_files]) and\
                any([cf for cf in LOG_FILENAMES if cf in log_video_files]):
              routes.add(route_name_prefix + '|' + m.groups()[0])
      route_names = list(routes)
      print(route_names)
    else:
      data_dir = os.path.dirname(args.data_dir)
      route_names = [os.path.basename(args.data_dir)]


  print('loading osm cache', args.lat, args.lon)
  query_results = load_results_from_file(args.lat, args.lon)
  if not query_results:
    print(args.lat, args.lon, ' cache file not found')
    sys.exit(0)

  context = zmq.Context()
  send_sock = context.socket(zmq.PUB)
  send_sock.bind("tcp://%s:%d" % ('*', 40472))
  recv_sock = context.socket(zmq.SUB)
  recv_sock.connect("tcp://%s:%d" % ('127.0.0.1', 40471))
  recv_sock.setsockopt(zmq.SUBSCRIBE, b"")

  ws_server = SimpleWebSocketServer('', 40471, SimpleEcho, selectInterval=0.)

  quiting = False
  for route_name in route_names:
    if quiting:
      break
    route = Route(route_name, data_dir)
    lr = MultiLogIterator(route.log_paths(), wraparound=False)
    _frame_id_lookup = {}
    _frame_reader = RouteFrameReader(
      route.camera_paths(), None, _frame_id_lookup, readahead=True)

    # TODO: Detect car from replay and use that to select carparams
    CP = ToyotaInterface.get_params("TOYOTA PRIUS 2017", {})
    VM = VehicleModel(CP)

    CalP = np.asarray(
      [[0, 0], [MODEL_INPUT_SIZE[0], 0], [MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]], [0, MODEL_INPUT_SIZE[1]]])
    vanishing_point = np.asarray([[MODEL_CX, MODEL_CY]])

    pygame.init()
    pygame.font.init()
    assert pygame_modules_have_loaded()

    size = (FULL_FRAME_SIZE[0] + UP.lidar_x, FULL_FRAME_SIZE[1])

    write_x = 5
    pygame.display.set_caption("openpilot debug UI")
    screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

    alert1_font = pygame.font.SysFont("arial", 30)
    alert2_font = pygame.font.SysFont("arial", 20)
    info_font = pygame.font.SysFont("arial", 15)

    camera_surface = pygame.surface.Surface(FULL_FRAME_SIZE, 0, 24).convert()
    cameraw_surface = pygame.surface.Surface(MODEL_INPUT_SIZE, 0, 24).convert()
    top_down_surface = pygame.surface.Surface((UP.lidar_x, UP.lidar_y), 0, 8)

    v_ego, angle_steers, angle_steers_des, angle_offset = 0., 0., 0., 0.
    enabled = False

    gas = 0.
    accel_override = 0.
    computer_gas = 0.
    brake = 0.
    steer_torque = 0.
    curvature = 0.
    computer_brake = 0.
    plan_source = 'none'
    long_control_state = 'none'
    a_ego = 0.0
    a_target = 0.0

    d_rel, y_rel, lead_status = 0., 0., False
    d_rel2, y_rel2, lead_status2 = 0., 0., False

    v_ego, v_pid, v_cruise, v_override = 0., 0., 0., 0.
    brake_lights = False

    alert_text1, alert_text2 = "", ""

    intrinsic_matrix = eon_intrinsics

    calibration = None
    img = np.zeros((FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3), dtype='uint8')
    imgw = np.zeros((160, 320, 3), dtype=np.uint8)  # warped image
    lid_overlay_blank = get_blank_lid_overlay(UP)
    img_offset = (0, 0)


    gps = None
    init_gps = None
    prev_gps_a = []
    traffic_signs_results = query_results['sign']
    traffic_signals_results = query_results['light']
    crossing_results = query_results['crossing']

    d_poly = None
    frame_msg = None
    live_track_msg = None
    model_data = None
    goto_next_route = False
    playing = True
    show_whitelines = (args.whiteline != 0)
    whiteline_vertical = False
    show_camera_view = True
    show_leads = False

    while not goto_next_route and not quiting:
      ws_server.serveonce()
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            quiting = True
            break  # break out of the for loop
          elif event.key == pygame.K_v:
            whiteline_vertical = not whiteline_vertical
            print('white line', whiteline_vertical)
          elif event.key == pygame.K_o:
            # Take snapshot of the full frame
            img_bgr = np.zeros_like(img)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, dst=img_bgr)
            save_path = '/home/nanami/data/{}-{}-{}.png'.format(args.route_name, lr._current_log, frame_id)
            cv2.imwrite(save_path, img_bgr)
          elif event.key == pygame.K_SPACE:
            playing = not playing
            print('Paused' if not playing else 'Playing...')
          elif event.key == pygame.K_t:
            args.tl = (1 if args.tl == 0 else 0)
            print('Traffic light', args.tl)
          elif event.key == pygame.K_n:
            goto_next_route = True
            print('Next route')
        elif event.type == pygame.QUIT:
          quiting = True
          break  # break out of the for loop
      if goto_next_route:
        break  # to break out of the while loop
      if not playing:
        time.sleep(0.2)
        continue
      try:
        msg = next(lr)
      except StopIteration:
        break
      smsg = msg.as_builder()
      typ = smsg.which()
      # Process msg
      if typ == 'carState':
        gas = smsg.carState.gas
        brake_lights = smsg.carState.brakeLights
        a_ego = smsg.carState.aEgo
        brake = smsg.carState.brake
        v_cruise = smsg.carState.cruiseState.speed
      elif typ == 'gpsLocation':
        for k, c in ws_server.connections.items():
          json_str = json.dumps({
            'src': 0,
            'lat': smsg.gpsLocation.latitude,
            'lng': smsg.gpsLocation.longitude,
            'acc': smsg.gpsLocation.accuracy});
          c.sendMessage(json_str)
      elif typ == 'gpsLocationExternal':
        #print('Lat', smsg.gpsLocationExternal.latitude, 'Lng', smsg.gpsLocationExternal.longitude, 'Acc', smsg.gpsLocationExternal.accuracy)
        if gps:
          prev_gps_a.append(gps)
          if len(prev_gps_a) > 30:
            prev_gps_a = prev_gps_a[1:]
        gps = smsg.gpsLocationExternal
        if not prev_gps_a:
          init_gps = gps
          print('Initial gps', gps.latitude, gps.longitude)
        for k, c in ws_server.connections.items():
          json_str = json.dumps({
            'src': 1,
            'lat': smsg.gpsLocationExternal.latitude,
            'lng': smsg.gpsLocationExternal.longitude,
            'acc': smsg.gpsLocationExternal.accuracy});
          c.sendMessage(json_str)
      elif typ == 'controlsState':
        v_ego = smsg.controlsState.vEgo
        angle_steers = smsg.controlsState.angleSteers
        model_bias = smsg.controlsState.angleModelBias
        curvature = smsg.controlsState.curvature
        v_pid = smsg.controlsState.vPid
        enabled = smsg.controlsState.enabled
        alert_text1 = smsg.controlsState.alertText1
        alert_text2 = smsg.controlsState.alertText2
        long_control_state = smsg.controlsState.longControlState
      elif typ == 'carControl':
        v_override = smsg.carControl.cruiseControl.speedOverride
        computer_brake = smsg.carControl.actuators.brake
        computer_gas = smsg.carControl.actuators.gas
        steer_torque = smsg.carControl.actuators.steer * 5.
        angle_steers_des = smsg.carControl.actuators.steerAngle
        accel_override = smsg.carControl.cruiseControl.accelOverride
      elif typ == 'plan':
        a_target = smsg.plan.aTarget
        plan_source = smsg.plan.longitudinalPlanSource
      elif typ == 'pathPlan':
        d_poly = np.array(smsg.pathPlan.dPoly)
      elif typ == 'model':
        model_data = extract_model_data(smsg)
      elif typ == 'liveTracks':
        live_track_msg = smsg
      elif typ == 'liveCalibration':
        calibration_message = smsg.liveCalibration
        calibration_message = smsg.liveCalibration
        extrinsic_matrix = np.asarray(calibration_message.extrinsicMatrix).reshape(3, 4)

        ke = intrinsic_matrix.dot(extrinsic_matrix)
        warp_matrix = get_camera_frame_from_model_frame(ke)

        calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)
      elif typ == 'radarState':
        d_rel = smsg.radarState.leadOne.dRel + RADAR_TO_CENTER
        y_rel = smsg.radarState.leadOne.yRel
        lead_status = smsg.radarState.leadOne.status
        d_rel2 = smsg.radarState.leadTwo.dRel + RADAR_TO_CENTER
        y_rel2 = smsg.radarState.leadTwo.yRel
        lead_status2 = smsg.radarState.leadTwo.status
      elif typ == 'frame':
        frame_msg = smsg
      elif typ == "encodeIdx" and smsg.encodeIdx.type == capnp_log.EncodeIndex.Type.fullHEVC:
        _frame_id_lookup[
          smsg.encodeIdx.frameId] = smsg.encodeIdx.segmentNum, smsg.encodeIdx.segmentId

      if typ in ['frame', 'encodeIdx'] and frame_msg:
        yuv_img = _frame_reader.get(frame_msg.frame.frameId, pix_fmt="yuv420p")
        if yuv_img is None:
          continue

        if not calibration:
          continue

        if False and lead_status and d_rel < 20.:
          continue

        transform = frame_msg.frame.transform
        frame_id = frame_msg.frame.frameId
        frame_msg = None
        yuv_img = yuv_img.tobytes()
        screen.fill((64, 64, 64))
        lid_overlay = lid_overlay_blank.copy()
        top_down = top_down_surface, lid_overlay

        if live_track_msg:
          maybe_update_radar_points(live_track_msg, top_down[1])

        if transform:
          yuv_transform = np.array(transform).reshape(3, 3)
        else:
          # assume frame is flipped
          yuv_transform = np.array([
            [-1.0, 0.0, FULL_FRAME_SIZE[0] - 1],
            [0.0, -1.0, FULL_FRAME_SIZE[1] - 1],
            [0.0, 0.0, 1.0]
          ])

        if yuv_img and len(yuv_img) == FULL_FRAME_SIZE[0] * FULL_FRAME_SIZE[1] * 3 // 2:
          yuv_np = np.frombuffer(yuv_img, dtype=np.uint8).reshape(FULL_FRAME_SIZE[1] * 3 // 2, -1)
          cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420, dst=img)
          cv2.warpAffine(img, np.dot(yuv_transform, calibration.model_to_full_frame)[:2],
                         (imgw.shape[1], imgw.shape[0]), dst=imgw, flags=cv2.WARP_INVERSE_MAP)
          img_bgr = np.zeros_like(imgw)
          cv2.cvtColor(imgw, cv2.COLOR_RGB2BGR, dst=img_bgr)
          yuv = np.zeros((240, 320), dtype=np.uint8)
          cv2.cvtColor(imgw, cv2.COLOR_RGB2YUV_I420, dst=yuv)
          if show_whitelines:
            frame_threshold, frame_foregnd = find_white_lines(img_bgr, yuv, whiteline_vertical)
          if args.save_img:
            img_bgr = np.zeros_like(imgw)
            cv2.cvtColor(imgw, cv2.COLOR_RGB2BGR, dst=img_bgr)
            save_path = '{}/{}-{}-{}.png'.format(args.save_img_dir, args.route_name, lr._current_log, frame_id)
            cv2.imwrite(save_path, img_bgr)
            #print('Image saved.', save_path)
            #save_img = False
            # don't render
            continue
          if model_data is not None:
            plot_model(model_data, VM, v_ego, curvature, imgw, calibration,
                       top_down, d_poly)
        else:
          img.fill(0)

        if show_leads:
          # draw red pt for lead car in the main img
          if lead_status:
            if calibration is not None:
              dx, dy = draw_lead_on(img, d_rel, y_rel, img_offset, calibration, color=(192, 0, 0))
            # draw red line for lead car
            draw_lead_car(d_rel, top_down)

          # draw red pt for lead car2 in the main img
          if lead_status2:
            if calibration is not None:
              dx2, dy2 = draw_lead_on(img, d_rel2, y_rel2, img_offset, calibration, color=(192, 0, 0))
            # draw red line for lead car
            draw_lead_car(d_rel2, top_down)
        if show_camera_view:
          if args.tl != 0:
            # x forward, y right, z up
            y = 0
            x = 100
            z = 10
            pt = calibration.car_project_to_screen(np.asarray([x, y, z, 1]))
            #l, r, t, b = int(pt[0]) - 150, int(pt[0]) + 150, int(pt[1]) - 150, int(pt[1]) + 150
            l, r, t, b = int(pt[0]) - 150, int(pt[0]) + 150, int(pt[1]) - 150, int(pt[1]) + 150
            cv2.rectangle(img, (l, t), (r, b), (255, 255, 255), 1)
            sub_img = img[t:b, l:r].copy()
            # Detect traffic light
            send_array(send_sock, sub_img)
            boxes = recv_array(recv_sock)
            if len(boxes) > 0:
              annotations = []
              for b in boxes:
                x1, y1, x2, y2 = b[1] + l, b[0] + t, b[3] + l, b[2] + t
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                annotations.append((b[1], b[0], b[3], b[2], 'tl'))
              img_name = '{}-{}.png'.format(route_name, frame_id)
              img_bgr = np.zeros_like(sub_img)
              cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR, dst=img_bgr)
              cv2.imwrite(os.path.join(args.out_imgs_dir, img_name), img_bgr)
              dict_to_xml(img_name, annotations,
                          os.path.join(args.out_annos_dir, os.path.splitext(img_name)[0] + '.xml'), sub_img)

          pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
          screen.blit(camera_surface, (0, 0))


        # display alerts
        alert_line1 = draw_transparent_text(alert1_font, alert_text1, (255, 0, 0))
        alert_line2 = draw_transparent_text(alert2_font, alert_text2, (255, 0, 0))
        screen.blit(alert_line1, (180, 150))
        screen.blit(alert_line2, (180, 190))

        cpw = warp_points(CalP, calibration.model_to_full_frame)
        vanishing_pointw = warp_points(vanishing_point, calibration.model_to_full_frame)
        posnet_box = np.array([[50, 237],
                               [1114, 237],
                               [1114, 637],
                               [50, 637]])
        pygame.draw.polygon(screen, BLUE, tuple(map(tuple, cpw)), 1)
        #pygame.draw.polygon(screen, RED, tuple(map(tuple, posnet_box)), 1)
        #pygame.draw.circle(screen, BLUE, map(int, map(round, vanishing_pointw[0])), 2)

        # x forward, y right, z up
        if True:
          for y in [-10, 0, 10]:
            prev_pt = None
            for x in [5, 6, 10, 25, 50, 200]:
              if x == 50 and y == 0:
                continue
              pt = calibration.car_project_to_screen(np.asarray([x, y, 0, 1]))
              text_line = draw_transparent_text(info_font, "{}m".format(x), YELLOW)
              screen.blit(text_line, (pt[0] + 10, pt[1] + 10))
              pygame.draw.circle(screen, YELLOW, map(int, map(round, (pt[0], pt[1]))), 2)
              if prev_pt:
                pygame.draw.line(screen, YELLOW, pt, prev_pt, 1)
              prev_pt = pt

        for y in [-5, 5]:
          for x in [50]:
            prev_pt = None
            for z in [0, 2, 3, 5]:
              pt = calibration.car_project_to_screen(np.asarray([x, y, z, 1]))
              if z != 0:
                text_line = draw_transparent_text(info_font, "{}m".format(z), YELLOW)
                screen.blit(text_line, (pt[0] + 10, pt[1]))
                pygame.draw.circle(screen, YELLOW, map(int, map(round, (pt[0], pt[1]))), 2)
              if prev_pt:
                pygame.draw.line(screen, YELLOW, pt, prev_pt, 1)
              prev_pt = pt


        pygame.surfarray.blit_array(cameraw_surface, imgw.swapaxes(0, 1))
        screen.blit(cameraw_surface, (0, 0))
        if show_whitelines:
          #rgb = np.zeros((160, 320, 3), dtype=np.uint8)
          #cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2RGB, dst=rgb)
          #pygame.surfarray.blit_array(cameraw_surface, rgb.swapaxes(0, 1))
          #screen.blit(cameraw_surface, (0, 160))
          pygame.surfarray.blit_array(cameraw_surface, frame_foregnd.swapaxes(0, 1))
          screen.blit(cameraw_surface, (320, 0))

        pygame.surfarray.blit_array(*top_down)
        screen.blit(top_down[0], (FULL_FRAME_SIZE[0], 0))

        write_y = 165

        # enabled
        enabled_line = draw_transparent_text(info_font, "ENABLED" if enabled else "DISABLED", GREEN if enabled else BLACK)
        screen.blit(enabled_line, (write_x, write_y))
        write_y += 30

        # brake lights
        brake_lights_line = draw_transparent_text(info_font, "BRAKE LIGHTS", RED if brake_lights else BLACK)
        screen.blit(brake_lights_line, (write_x, write_y))
        write_y += 30

        # speed
        kmph = v_ego / 1000. * 60. * 60.
        meters = meters_osm = 0.
        if gps and prev_gps_a:
          meters = lat_lon_distance(gps.latitude, gps.longitude, gps.altitude, init_gps.latitude, init_gps.longitude, init_gps.altitude)
          meters_osm = lat_lon_distance(gps.latitude, gps.longitude, gps.altitude, args.lat, args.lon, gps.altitude)
          v_ego_line = draw_transparent_text(
            info_font, "DIST: {}km, {}km, SPEED: {}kmph, ".format(round(meters/1000., 1), round(meters_osm/1000., 1), round(kmph, 1)), YELLOW)
          screen.blit(v_ego_line, (write_x, write_y))
          write_y += 30

        if gps:
          v_ego_line = draw_transparent_text(
            info_font, "LAT: {}, LNG: {}, ACC: {}".format(gps.latitude, gps.longitude, gps.accuracy), YELLOW)
          screen.blit(v_ego_line, (write_x, write_y))
          write_y += 30

        if lead_status:
          text_line = draw_transparent_text(info_font, "Lead dist: {}, y: {}".format(int(d_rel), int(y_rel)), YELLOW)
          screen.blit(text_line, (write_x, write_y))
          write_y += 30
        if lead_status2:
          text_line = draw_transparent_text(info_font, "Lead2 dist: {}, y: {}".format(int(d_rel2), int(y_rel2)), YELLOW)
          screen.blit(text_line, (write_x, write_y))
          write_y += 30

        # angle offset
        model_bias_line = info_font.render("MODEL BIAS: " + str(round(model_bias, 2)) + " deg", True, YELLOW)
        screen.blit(model_bias_line, (write_x, write_y))
        write_y += 30

        angle_offset_line = draw_transparent_text(info_font, "STEER OFFSET: " + str(round(angle_offset, 2)) + " deg", YELLOW)
        screen.blit(angle_offset_line, (write_x, write_y))
        write_y += 30

        # long control state
        long_control_state_line = draw_transparent_text(info_font, "LONG CONTROL STATE: " + str(long_control_state), YELLOW)
        screen.blit(long_control_state_line, (write_x, write_y))
        write_y += 30

        # long mpc source
        text_line = draw_transparent_text(info_font, "LONG MPC SOURCE: " + str(plan_source), YELLOW)
        screen.blit(text_line, (write_x, write_y))
        write_y += 30

        if gps and prev_gps_a and False:
          pt = (MAP_VIEW_LEFT + MAP_VIEW_WIDTH_HEIGHT // 2, MAP_VIEW_TOP + MAP_VIEW_WIDTH_HEIGHT // 2)
          pygame.draw.circle(screen, BLUE, map(int, map(round, pt)), MAP_VIEW_WIDTH_HEIGHT / 2, 1)
          text_line = draw_transparent_text(info_font, 'N', YELLOW)
          screen.blit(text_line, (MAP_VIEW_LEFT + MAP_VIEW_WIDTH_HEIGHT // 2 - 5, MAP_VIEW_TOP + 5))
          text_line = draw_transparent_text(info_font, 'S', YELLOW)
          screen.blit(text_line, (MAP_VIEW_LEFT + MAP_VIEW_WIDTH_HEIGHT // 2 - 5, MAP_VIEW_TOP + MAP_VIEW_WIDTH_HEIGHT - 25))

          l = lambda lat, lon: lat_lon_to_screen_x_y(lat, lon,
                                     MAP_VIEW_LEFT, MAP_VIEW_TOP, MAP_VIEW_LEFT + MAP_VIEW_WIDTH_HEIGHT, MAP_VIEW_TOP + MAP_VIEW_WIDTH_HEIGHT,
                                     gps.latitude, gps.longitude, max_dist=MAX_DISTANCE)
          pt = l(gps.latitude, gps.longitude)
          pygame.draw.circle(screen, BLUE, map(int, map(round, pt)), 2)
          pygame.draw.line(screen, GREEN, l(gps.latitude, gps.longitude), l(prev_gps_a[0].latitude, prev_gps_a[0].longitude), 1)

          nodes = find_closest_nodes(
            traffic_signs_results, gps.latitude, gps.longitude, gps.altitude, prev_gps_a[0].latitude, prev_gps_a[0].longitude, max_bearing_offset=90., max_dist=MAX_DISTANCE)
          for d, bf, node in nodes:
            pt = l(float(node.lat), float(node.lon))
            pygame.draw.circle(screen, GREEN, map(int, map(round, pt)), 2)
            s = "Traffic sign: {}, {} meters, {} degrees".format(parse_traffic_sign(node), int(d), int(bf))
            text_line = draw_transparent_text(info_font, s, YELLOW)
            screen.blit(text_line, (write_x, write_y))
            write_y += 30

          nodes = find_closest_nodes(
            traffic_signals_results, gps.latitude, gps.longitude, gps.altitude, prev_gps_a[0].latitude, prev_gps_a[0].longitude, max_bearing_offset=90., max_dist=MAX_DISTANCE)
          for d, bf, node in nodes:
            pt = l(float(node.lat), float(node.lon))
            pygame.draw.circle(screen, RED, map(int, map(round, pt)), 2)
            pygame.draw.circle(screen, GREEN, map(int, map(round, (pt[0] + 5, pt[1]))), 2)
            box = np.array([[pt[0] - 6, pt[1] - 6],
                                   [pt[0] + 10, pt[1] - 6],
                                   [pt[0] + 10, pt[1] + 5],
                                   [pt[0] - 6, pt[1] + 5 ]])
            pygame.draw.polygon(screen, BLUE, tuple(map(tuple, box)), 1)
            s = "Traffic light: {}, {} meters, {} degrees".format(parse_traffic_signal(node), int(d), int(bf))
            text_line = draw_transparent_text(info_font, s, YELLOW)
            screen.blit(text_line, (write_x, write_y))
            write_y += 30
            car_frame_pt_upper = np.array([d * math.cos(math.radians(bf)), -d * math.sin(math.radians(bf)), 4, 1])
            car_frame_pt_lower = np.array([d * math.cos(math.radians(bf)), -d * math.sin(math.radians(bf)), 2, 1])
            pt1 = calibration.car_to_screen_3d.dot(car_frame_pt_upper)
            pt2 = calibration.car_to_screen_3d.dot(car_frame_pt_lower)
            box = np.array([[pt1[0]/pt1[2] - 20, pt1[1]/pt1[2]],
                                   [pt1[0]/pt1[2] + 20, pt1[1]/pt1[2]],
                                   [pt2[0]/pt2[2] + 20, pt2[1]/pt2[2]],
                                   [pt2[0]/pt2[2] - 20, pt2[1]/pt2[2]]])
            pygame.draw.polygon(screen, BLUE, tuple(map(tuple, box)), 1)
            text_line = draw_transparent_text(alert1_font, "{}".format(node.id), YELLOW)
            screen.blit(text_line, (pt1[0]/pt1[2], pt1[1]/pt1[2]))

          nodes = find_closest_nodes(
            crossing_results, gps.latitude, gps.longitude, gps.altitude, prev_gps_a[0].latitude, prev_gps_a[0].longitude, max_bearing_offset=90., max_dist=MAX_DISTANCE)
          for d, bf, node in nodes:
            pt = l(float(node.lat), float(node.lon))
            pygame.draw.circle(screen, YELLOW, map(int, map(round, pt)), 2)
            s = "Crossing: {}, {} meters, {} degrees".format(parse_crossing(node), int(d), int(bf))
            text_line = draw_transparent_text(info_font, s, YELLOW)
            screen.blit(text_line, (write_x, write_y))
            write_y += 30

        # this takes time...vsync or something
        pygame.display.flip()


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
