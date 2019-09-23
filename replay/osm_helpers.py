import overpy
import numpy as np
from common.transformations.coordinates import geodetic2ecef, ecef2geodetic
from scipy import spatial
import argparse
import sys
import os
import math
from decimal import Decimal

OVERPASS_API_URL = "https://overpass.kumi.systems/api/interpreter"

TAGS = {
  'sign': 'traffic_sign',
  'light': 'highway=traffic_signals',
  'crossing': 'highway=crossing',
}

CACHE_DIR = os.path.expanduser("~/.osm_helper_cache")


def mkdirs_exists_ok(path):
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

def get_cache_file_path(lat, lon, k):
  mkdirs_exists_ok(CACHE_DIR)
  return os.path.join(CACHE_DIR, '{},{}.{}.json'.format(lat, lon, k))


def init_overpass_api():
  return overpy.Overpass(url=OVERPASS_API_URL)


def lat_lon_distance(lat1, lon1, alt1, lat2, lon2, alt2):
  ecef1 = geodetic2ecef((lat1, lon1, alt1))
  ecef2 = geodetic2ecef((lat2, lon2, alt2))
  return np.linalg.norm(ecef1 - ecef2)


# Builds a query to find all nodes with specific tag
def build_node_query(lat, lon, tag, radius):
  pos = "  (around:%f,%f,%f)" % (radius, lat, lon)
  lat_lon = "(%f,%f)" % (lat, lon)
  q = """
  [out:json];
  node
  """ + pos + """
  [
  """ + tag + """
  ];
  out;
  """
  return q


def parse_query_result(result):
  nodes = []
  real_nodes = []

  for n in result.nodes:
    nodes.append((float(n.lat), float(n.lon), 0))
    real_nodes.append(n)

  if len(nodes) <= 0:
    return None, None
  nodes = np.asarray(nodes)
  nodes = geodetic2ecef(nodes)
  tree = spatial.cKDTree(nodes)
  return tree, real_nodes

# tag examples:
#   highway=traffic_signals
#   highway=crossings
#   traffic_signs
# radius is in meters
def query_nodes(api, lon, lat, tag='traffic_sign', radius=1000):
  q = build_node_query(lat, lon, tag, radius=radius)
  print(q)
  result = api.query(q)
  if len(result.nodes) == 0:
    print(tag, radius, 'not found by OSM')
    return None, None
  return parse_query_result(result)


def calculate_initial_compass_bearing(pointA, pointB):
  lat1 = math.radians(pointA[0])
  lat2 = math.radians(pointB[0])

  diffLong = math.radians(pointB[1] - pointA[1])

  x = math.sin(diffLong) * math.cos(lat2)
  y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                         * math.cos(lat2) * math.cos(diffLong))

  initial_bearing = math.atan2(x, y)
  initial_bearing = math.degrees(initial_bearing)
  compass_bearing = (initial_bearing + 360) % 360
  return compass_bearing


# Fine all nodes within max_dist meters, or the nearest node
def find_closest_nodes(query_results, lat, lon, alt, prev_lat, prev_lon, max_bearing_offset=15., max_dist=200.):
  tree, real_nodes = query_results
  if not tree:
    return
  cur_pos = geodetic2ecef((lat, lon, 0))
  nodes = tree.query_ball_point(cur_pos, max_dist)
  if not nodes and False:
    nodes = [tree.query(cur_pos)[1]]
  r = []
  for n in nodes:
    node = real_nodes[n]
    b1 = calculate_initial_compass_bearing((Decimal(prev_lat), Decimal(prev_lon)), (Decimal(lat), Decimal(lon)))
    b2 = calculate_initial_compass_bearing((Decimal(lat), Decimal(lon)), (node.lat, node.lon))
    bf = abs(b2 - b1)
    if bf < max_bearing_offset:
      d = lat_lon_distance(lat, lon, alt, float(node.lat), float(node.lon), alt)
      r.append((d, b2 - b1, node))
  return r

def get_mercator_x_y(width , height, lat, lon):
  lat_rad = lat * math.pi / 180.0
  merc = 0.5 * math.log((1 + math.sin(lat_rad)) / (1 - math.sin(lat_rad)))
  return (int(round(math.fmod((width * (180.0 + lon) / 360.0), (1.5 * width)))),
         int(round((height / 2) - (width * merc / (2 * math.pi)))))

# mercator projection
def lat_lon_to_screen_x_y(lat, lon, x0, y0, x1, y1, center_lat, center_lon, max_dist=100.):
  equaltor_len = 40075. * 1000.
  polar_diameter = 12713.6 * 1000.
  # pixel size required for projecting the whole earth
  width = equaltor_len / (max_dist * 2) * (x1 - x0)
  height = polar_diameter / (max_dist * 2) * (y1 - y0)

  c_merc_x, c_merc_y = get_mercator_x_y(width, height, center_lat, center_lon)
  merc_x, merc_y = get_mercator_x_y(width, height, lat, lon)
  return x0 + (x1 - x0) / 2 + merc_x - c_merc_x, y0 + (y1 - y0) / 2 + merc_y - c_merc_y

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY2:
    from urllib2 import urlopen
    from urllib2 import HTTPError
elif PY3:
    from urllib.request import urlopen
    from urllib.error import HTTPError

# overpy doesn't expose api response data, so we clone the api call code
def call_overpass_api(query):
  if not isinstance(query, bytes):
    query = query.encode("utf-8")

  f = urlopen(OVERPASS_API_URL, query)
  read_chunk_size = 4096
  response = f.read(read_chunk_size)
  while True:
    data = f.read(read_chunk_size)
    if len(data) == 0:
      break
    response = response + data
  f.close()
  return response


def load_results_from_file(lat, lon):
  api = init_overpass_api()
  results_dict = {}
  for k, tag in TAGS.items():
    f = get_cache_file_path(lat, lon, k)
    if not os.path.exists(f):
      return None
    with open(f, 'r') as f:
      d = f.read()
      results_dict[k] = parse_query_result(api.parse_json(d))
  return results_dict

def parse_traffic_signal(node):
  tags = node.tags
  r = ''
  if 'traffic_signals' in tags:
    r += 'type: {}, '.format(tags['traffic_signals'])
  if 'traffic_signals:direction' in tags:
    r += 'dir: {}, '.format(tags['traffic_signals:direction'])
  return r

def parse_crossing(node):
  tags = node.tags
  r = ''
  if 'crossing' in tags:
    r += 'type: {}, '.format(tags['crossing'])
  return r

def parse_traffic_sign(node):
  tags = node.tags
  r = ''
  if 'traffic_sign' in tags:
    r += 'type: {}, '.format(tags['traffic_sign'])
  if 'direction' in tags:
    r += 'dir: {}, '.format(tags['direction'])
  return r

def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Query map data from OpenStreetMap.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("lat", nargs="?", type=float, default=37.721674199999995,
                      help="Latitude")
  parser.add_argument("lon", nargs="?", type=float, default=-122.4691069,
                      help="Longitude")
  parser.add_argument("--radius", nargs="?", type=float, default=50000.,
                      help="Longitude")
  parser.add_argument("--test", nargs="?", default=False,
                      help="Read cached data and dump out")
  return parser


def query_and_cache_results(lat, lon, radius):
  results_json_dict = {}
  for k, tag in TAGS.items():
    q = build_node_query(lat, lon, tag, radius)
    print(q)
    print('Querying {}...'.format(tag))
    d = call_overpass_api(q)
    with open(get_cache_file_path(args.lat, args.lon, k), 'w') as f:
      f.write(d.decode('utf-8'))
  print('Done.')


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])

  if not args.test:
    query_and_cache_results(args.lat, args.lon, args.radius)
  else:
    results_dict = load_results_from_file(args.lat, args.lon)
    for k, tag in TAGS.items():
      tree, real_nodes = results_dict[k]
      print(tag, 'nodes', len(real_nodes))