import os
import numpy as np
import scipy.sparse as spsp
import networkx as nx
from networkx.algorithms.community import kernighan_lin

def kl_2partition(coo_adj, outfolder=None):
  """
  Params:
    coo_adj: scipy.sparse.coo_matrix
    outfolder: generating as 'adj_2p1.npz', 'adj_2p2.npz'
  Return:
    sub coo adj list,
    sub node idx to upper graph idx conversion array list
  """
  # functions for selecting idx
  def select_partition(orig, partition):
    return orig in partition
  vfunc = np.vectorize(select_partition, excluded=['partition'])

  g = nx.from_scipy_sparse_matrix(coo_adj)
  src_node = coo_adj.row
  dst_node = coo_adj.col
  partitions = kernighan_lin.kernighan_lin_bisection(g, max_iter=40)

  subid2fullid = []
  sub_coo_adjs = []
  for p in partitions:
    p = np.array(list(p), dtype=np.int64)
    # select the idx in common to build sub graph
    src_edge = vfunc(orig=src_node, partition=p)
    dst_edge = vfunc(orig=dst_node, partition=p)
    p_idx = src_edge * dst_edge 
    # get pair with idx in full graph
    p_src, p_dst = src_node[p_idx], dst_node[p_idx]
    edge = np.ones(len(p_src), dtype=np.int)
    # convert idx into new idx start from 0
    subid2fullid.append(p)
    fullid2subid = np.zeros(np.max(p) + 1, dtype=np.int64)
    fullid2subid[p] = np.arange(len(p))
    p_src, p_dst = fullid2subid[p_src], fullid2subid[p_dst]
    del fullid2subid
    # create sub coo adj
    sub_coo_adj = spsp.coo_matrix((edge, (p_src, p_dst)), shape=(len(p),len(p)))
    sub_coo_adjs.append(sub_coo_adj)
  if outfolder is not None:
    for idx, (subadj, sub2full) in enumerate(zip(sub_coo_adjs, subid2fullid)):
      adjpath = os.path.join(outfolder, 'adj_2p{}.npz'.format(idx+1))
      mappath = os.path.join(outfolder, 'map_2p{}.npy'.format(idx+1))
      spsp.save_npz(adjpath, subadj)
      np.save(mappath, sub2full)
  return sub_coo_adjs, subid2fullid


def test(draw=False):
  # create a 12 node graph
  src_node = np.array(
    [0,0,0,1,1,1,1,2,2,3,3,3,3,4,4,4,5,5,5,6,6,7,7,8,8,9,9,9,10,10,10,11],
    dtype=np.int
  )
  dst_node = np.array(
    [9,10,1,9,0,8,2,1,3,8,2,4,6,3,7,5,5,6,7,5,3,4,5,1,3,0,1,10,0,9,11,10],
    dtype=np.int
  )
  edge = np.ones(len(src_node), dtype=np.int)
  coo_adj = spsp.coo_matrix((edge, (src_node, dst_node)), shape=(12,12))
  # partition
  subadjs, maps = kl_2partition(coo_adj)
  if draw:
    for subadj, sub2full in zip(subadjs, maps):
      labels = {idx: sub2full[idx] for idx in range(len(sub2full))}
      subg = nx.from_scipy_sparse_matrix(subadj)
      pos = nx.kamada_kawai_layout(subg)
      nx.draw(subg, pos, node_color=[[.7, .7, .7]])
      nx.draw_networkx_labels(subg, pos, labels=labels)
      break


if __name__ == '__main__':
  test(draw=True)

    