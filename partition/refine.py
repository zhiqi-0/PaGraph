import os
import numpy as np
import scipy.sparse as spsp
import networkx as nx

from .utils import *


def build_train_graph(coo_adj, train_nids, num_hop):
  """
  Build training graphs
  Params:
    coo_adj: coo sparse adjacancy matrix for sub graph
    sub2fullid: np array mapping sub_adj node idx to full graph node idx
    train_nids: np array for training node idx.
  Returns:
    coo_adj: coo sparse adjacancy matrix for new graph
    sub2fullid: new mappings for new sub graph idx to full graph node idx
  """
  # step 1: get in-neighbors for each train nids
  neighbors = get_num_hop_in_neighbors(coo_adj, train_nids, num_hop)
  # step 2: get edge (src, dst) pair
  isin_mask_vfunc = np.vectorize(include, excluded=['node_range'])
  src = coo_adj.row
  dst = coo_adj.col
  neighbors = [train_nids] + neighbors
  train_src = []
  train_dst = []
  for hop in range(num_hop):
    hop_dst = neighbors[hop]
    hop_src = neighbors[hop+1]
    src_mask = isin_mask_vfunc(nid=src, node_range=hop_src)
    dst_mask = isin_mask_vfunc(nid=dst, node_range=hop_dst)
    mask = src_mask * dst_mask
    hop_src = src[mask]
    hop_dst = dst[mask]
    train_src.append(hop_src)
    train_dst.append(hop_dst)
  train_src = np.concatenate(tuple(train_src))
  train_dst = np.concatenate(tuple(train_dst))
  # step 3: translate src, dst node ids to new namespace
  sub2fullid = np.unique(np.concatenate((train_src, train_dst)))
  train_sub_src = full2sub_nid(sub2fullid, train_src)
  train_sub_dst = full2sub_nid(sub2fullid, train_dst)
  # step 4: build graph
  edge = np.ones(len(train_src), dtype=np.int)
  new_coo_adj = spsp.coo_matrix((edge, (train_sub_src, train_sub_dst)),
                                shape=(len(train_src, len(train_src))))
  return new_coo_adj, sub2fullid
  

def wrap_neighbor(full_adj, sub_adj, sub2fullid, num_hop, train_nids=None):
  """
  Params:
    full_adj: coo sparse adjacancy matrix for full graph
    sub_adj:  coo sparse adjacancy matrix for sub graph
    sub2fullid: np array mapping sub_adj node idx to full graph node idx
    num_hop: num-hop neighbors for each node will be included in the sub graph
    train_nids: np array for training node idx. If provided, only training node
                neighbors will be included.
  Returns:
    sub_adj: coo sparse adjacancy matrix for wrapped sub graph
    sub2fullid: New mappings for new sub graph idx to full graph node idx
  """
  # step 1: get extra edge tuple (src, dst)
  isin_mask_vfunc = np.vectorize(include, excluded=['node_range'])
  sub_train_nids_infull_mask = isin_mask_vfunc(nid=sub2fullid, node_range=train_nids)
  sub_train_nids_infull = sub2fullid[sub_train_nids_infull_mask]
  nodes = sub_train_nids_infull
  neighbors = get_num_hop_in_nodes(full_adj, nodes, num_hop,
                                   excluded_nodes=sub2fullid)
  neighbors = [nodes] + neighbors
  extra_src = []
  extra_dst = []
  full_src = full_adj.row
  full_dst = full_adj.col
  for hop in range(num_hop):
    # only in-edge will participate into computations
    dst = neighbors[hop]
    src = neighbors[hop + 1]
    extra_src_mask = isin_mask_vfunc(nid=full_src, node_range=src)
    extra_dst_mask = isin_mask_vfunc(nid=full_dst, node_range=dst)
    extra_mask = extra_src_mask * extra_dst_mask
    extra_src.append(full_src[extra_mask])
    extra_dst.append(full_dst[extra_mask])
  extra_src = np.concatenate(tuple(extra_src))
  extra_dst = np.concatenate(tuple(extra_dst))
  # step 2: translate extra node id, edges into sub graph namespace
  new_src = np.concatenate((sub2fullid[sub_adj.row], extra_src))
  new_dst = np.concatenate((sub2fullid[sub_adj.col], extra_dst))
  sub2fullid = np.unique(np.concatenate((new_src, new_dst)))
  new_sub_src = full2sub_nid(sub2fullid, new_src)
  new_sub_dst = full2sub_nid(sub2fullid, new_dst)
  # step 3: construct new sub graph
  edge = np.ones(len(new_src), dtype=np.int)
  new_coo_adj = spsp.coo_matrix((edge, (new_sub_src, new_sub_dst)),
                                shape=(len(new_src, len(new_src))))
  return new_coo_adj, full2subid
  

def exclude(nid, node_range):
  return not nid in node_range

def include(nid, node_range):
  return nid in node_range
