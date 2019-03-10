import overpy
import numpy as np
from common.transformations.coordinates import geodetic2ecef
from scipy import spatial

OVERPASS_API_URL = "https://overpass.kumi.systems/api/interpreter"


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
  node
  """ + pos + """
  [
  """ + tag + """
  ];
  out;
  """
  return q

# tag examples:
#   highway=traffic_signals
#   highway=crossings
#   traffic_signs
# radius is in meters
def query_nodes(api, lon, lat, tag='traffic_sign', radius=1000):
  q = build_node_query(lat, lon, tag, radius=radius)
  print(q)
  new_result = api.query(q)
  if len(new_result.nodes) == 0:
    print(tag, radius, 'not found by OSM')
    return None, None, None
  nodes = []
  real_nodes = []

  for n in new_result.nodes:
    nodes.append((float(n.lat), float(n.lon), 0))
    real_nodes.append(n)

  nodes = np.asarray(nodes)
  nodes = geodetic2ecef(nodes)
  tree = spatial.cKDTree(nodes)
  return tree, nodes, real_nodes

def find_closest_node(query_results, lat, lon, alt):
  tree, nodes, real_nodes = query_results
  if not tree:
    return
  cur_pos = geodetic2ecef((lat, lon, 0))
  #nodes = tree.query_ball_point(cur_pos, 500)
  n = tree.query(cur_pos)[1]
  real_node = real_nodes[n]
  d = lat_lon_distance(lat, lon, alt, float(real_node.lat), float(real_node.lon), alt)
  if 'highway' in real_node.tags and real_node.tags['highway'] == 'traffic_signals':
    print(d, 'meters', 'from nearest traffic signal', real_node.lat, real_node.lon)
  if 'traffic_sign' in real_node.tags:
    print(d, 'meters', 'from nearest traffic sign ', real_node.tags['traffic_sign'],  real_node.lat, real_node.lon)