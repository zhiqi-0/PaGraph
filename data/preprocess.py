"""
Preprocess dataset to fit the input
"""

import numpy as np

def pp2adj(filepath, is_direct=True, delimiter=' ',
           outfile=None):
  """
  Convert (vertex vertex) tuple into numpy adj matrix
  If vnum is not provided, vnum := max(vid) - min(vid) + 1
  Similar to enum.
  adj matrix will be returned.
  If outfile is provided, also save it.
  """
  pp = np.loadtxt(filepath, delimiter=delimiter)
  src_node = np.int(pp[0,:])
  dst_node = np.int(pp[1,:])
  max_nid = max(np.max(src_node), np.max(dst_node))
  min_nid = min(np.min(src_node), np.min(dst_node))
  vnum = max_nid - min_nid + 1
  enum = len(src_node)
  # scale node id from 0
  src_node -= min_nid
  dst_node -= min_nid
  print('vertex#: {} edge#: {}'.format(vnum, enum))
  # create numpy adj matrix
  adj = np.int(np.zeros((vnum, vnum)))
  adj[src_node, dst_node] = 1
  if not is_direct:
    adj[dst_node, src_node] = 1
  # output to file
  if outfile is not None:
    np.save(outfile, adj)
  return adj


def random_feature(vnum, feat_size, outfile=None):
  """
  Generate random features using numpy
  Params:
    vnum:       feature num (aka. vertex num)
    feat_size:  feature dimension 
    outfile:    save to the file if provided
  Returns:
    numpy array obj with shape of [vnum, feat_size]
  """
  feat_mat = np.float32(np.random((vnum, feat_size)))
  if outfile:
    np.save(outfile, feat_mat)
  return feat_mat
