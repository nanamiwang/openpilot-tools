import os
import re
from glob import glob
from collections import defaultdict

SEGMENT_NAME_RE = r'[a-z0-9]{16}[|_][0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2}--[0-9]+'
EXPLORER_FILE_RE = r'^({})--([a-z]+\.[a-z0-9]+)$'.format(SEGMENT_NAME_RE)
OP_SEGMENT_DIR_RE = r'^({})$'.format(SEGMENT_NAME_RE)
RELATIVE_SEGMENT_NAME_RE = r'[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2}--[0-9]+'

LOG_FILENAMES = ['rlog.bz2', 'raw_log.bz2', 'log2.gz', 'ilog.7z']
CAMERA_FILENAMES = ['fcamera.hevc', 'video.hevc', 'acamera', 'icamera']

class Route(object):
  def __init__(self, route_name, data_dir):
    self.route_name = route_name
    if not os.path.isdir(os.path.dirname(self.route_name)):
      self.route_name = self.route_name.replace('_', '|')
    self._segments = self._get_segments(data_dir)

  @property
  def segments(self):
    return self._segments

  def log_paths(self):
    max_seg_number = self._segments[-1].canonical_name.segment_num
    log_path_by_seg_num = {s.canonical_name.segment_num: s.log_path for s in self._segments}
    return [log_path_by_seg_num.get(i, None) for i in range(max_seg_number+1)]

  def camera_paths(self):
    max_seg_number = self._segments[-1].canonical_name.segment_num
    camera_path_by_seg_num = {s.canonical_name.segment_num: s.camera_path for s in self._segments}
    return [camera_path_by_seg_num.get(i, None) for i in range(max_seg_number+1)]

  def _get_segments(self, data_dir):
    if not data_dir:
      #self.route_name is /home/nanami/data/my_logs/55b38f438c826c9a/2019-08-13--07-12-52
      data_dir = os.path.dirname(self.route_name)
      self.route_name = os.path.basename(self.route_name)

    files = os.listdir(data_dir)
    segment_files = defaultdict(list)

    for f in files:
      fullpath = os.path.join(data_dir, f)
      explorer_match = re.match(EXPLORER_FILE_RE, f)
      op_match = re.match(OP_SEGMENT_DIR_RE, f)
      relative_segment_name_match = re.match(RELATIVE_SEGMENT_NAME_RE, f)

      if explorer_match:
        segment_name, fn = explorer_match.groups()
        if segment_name.replace('_', '|').startswith(self.route_name):
          segment_files[segment_name].append((fullpath, fn))
      elif op_match and os.path.isdir(fullpath):
        segment_name, = op_match.groups()
        if segment_name.startswith(self.route_name):
          for seg_f in os.listdir(fullpath):
            segment_files[segment_name].append((os.path.join(fullpath, seg_f), seg_f))
      elif f == self.route_name:
        for seg_num in os.listdir(fullpath):
          if not seg_num.isdigit():
            continue

          segment_name = '{}--{}'.format(self.route_name, seg_num)
          for seg_f in os.listdir(os.path.join(fullpath, seg_num)):
            segment_files[segment_name].append((os.path.join(fullpath, seg_num, seg_f), seg_f))
      elif relative_segment_name_match and os.path.isdir(fullpath):
        if f.startswith(self.route_name):
          for ext in ('*.hevc', '*.bz2'):
            # 0.6.2 logs
            fpaths = glob(os.path.join(fullpath, '*', ext))
            for fpath in fpaths:
              if os.path.isfile(fpath):
                segment_files[f].append((fpath, os.path.basename(fpath)))
            # 0.5 logs
            fpaths = glob(os.path.join(fullpath, ext))
            for fpath in fpaths:
              if os.path.isfile(fpath):
                segment_files[f].append((fpath, os.path.basename(fpath)))

    segments = []
    for segment, files in segment_files.items():
      try:
        log_path = next(path for path, filename in files if filename in LOG_FILENAMES)
      except StopIteration:
        log_path = None

      try:
        camera_path = next(path for path, filename in files if filename in CAMERA_FILENAMES)
      except StopIteration:
        camera_path = None

      segments.append(RouteSegment(segment, log_path, camera_path))

    if len(segments) == 0:
      raise ValueError('Could not find segments for route {} in data directory {}'.format(self.route_name, data_dir))
    return sorted(segments, key=lambda seg: seg.canonical_name.segment_num)

class RouteSegment(object):
  def __init__(self, name, log_path, camera_path):
    self._name = RouteSegmentName(name)
    self.log_path = log_path
    self.camera_path = camera_path

  @property
  def name(self): return str(self._name)

  @property
  def canonical_name(self): return self._name

class RouteSegmentName(object):
  def __init__(self, name_str):
    self._segment_name_str = name_str
    self._route_name_str, num_str = self._segment_name_str.rsplit("--", 1)
    self._num = int(num_str)

  @property
  def segment_num(self): return self._num

  def __str__(self): return self._segment_name_str
