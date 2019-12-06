import os
import argparse
import numpy as np
import scipy.sparse as spsp
import networkx as nx
from networkx.algorithms.community import kernighan_lin

import refine
from utils import *

def kl_2partition(coo_adj, outfolder=None):
  """
  Params:
    coo_adj: scipy.sparse.coo_matrix
    outfolder: generating as 'adj_2p1.npz', 'adj_2p2.npz'
  Return:
    sub coo adj list,
    sub node idx to upper graph idx conversion array list
  """
  isin_mask_vfunc = np.vectorize(include, excluded=['node_range'])

  g = nx.from_scipy_sparse_matrix(coo_adj)
  src_node = coo_adj.row
  dst_node = coo_adj.col
  partitions = kernighan_lin.kernighan_lin_bisection(g, max_iter=40)

  subid2fullid = []
  sub_coo_adjs = []
  for p in partitions:
    p = np.array(list(p), dtype=np.int64)
    # select the idx in common to build sub graph
    src_edge = isin_mask_vfunc(nid=src_node, node_range=p)
    dst_edge = isin_mask_vfunc(nid=dst_node, node_range=p)
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
  parser = argparse.ArgumentParser(description='Partition')

  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")
  
  parser.add_argument("--train-graph", dest='train_graph', action='store_true')
  parser.set_defaults(train_graph=False)

  parser.add_argument("--partition", type=int, default=2,
                      help="num of partitions")
  
  parser.add_argument("--wrap-neighbor", dest='wrap_neighbor', action='store_true')
  parser.set_defaults(wrap_neighbor=False)

  parser.add_argument("--num-hop", type=int, default=1,
                      help="num of hop neighbors required for a batch")
  
  args = parser.parse_args()

  if not os.path.exists(args.dataset):
    print('{}: No such a dataset folder'.format(args.dataset))
    sys.exit(-1)
  train_dataset = os.path.join(args.dataset, 'train')
  partition_dataset = os.path.join(args.dataset, 'partition')
  try:
    os.mkdir(train_dataset)
  except FileExistsError:
    pass
  try:
    os.mkdir(partition_dataset)
  except FileExistsError:
    pass

  # original dataset path
  adj_file = os.path.join(args.dataset, 'adj.npz')
  feat_file = os.path.join(args.dataset, 'feat.npy')
  mask_file = os.path.join(args.dataset, 'train.npy')
  label_file = os.path.join(args.dataset, 'labels.npy')
  # train dataset path
  train_adj_file = os.path.join(train_dataset, 'adj.npz')
  train_feat_file = os.path.join(train_dataset, 'feat.npy')
  train_mask_file = os.path.join(train_dataset, 'train.npy')
  train_label_file = os.path.join(train_dataset, 'labels.npy')
  train2fullid_file = os.path.join(train_dataset, 'train2fullid.npy')

  # generate train graph
  if args.train_graph:
    if os.path.exists(train_adj_file):
      adj = spsp.load_npz(train_adj_file)
      mask = np.load(mask_file).astype(np.bool)
      train_nids = np.arange(adj.shape[0])[mask]
    else:
      adj = spsp.load_npz(adj_file)
      feat = np.load(feat_file)
      mask = np.load(mask_file).astype(np.bool)
      label = np.load(label_file)
      nids = np.arange(adj.shape[0])[mask]
      adj, train2fullid = refine.build_train_graph(adj, nids, args.num_hop)
      train_feat = feat[train2fullid]
      train_label = label[train2fullid]
      train_mask = mask[train2fullid]
      train_nids = np.arange(adj.shape[0])[train_mask.astype(np.bool)]
      # save
      spsp.save_npz(train_adj_file, adj)
      np.save(train_feat_file, train_feat)
      np.save(train_label_file, train_label)
      np.save(train_mask_file, train_mask)
      np.save(train2fullid_file, train2fullid)
  else:
    adj = spsp.load(adj_file)
    train_mask = np.load(mask_file)
    train_nids = np.arange(adj_file.shape[0])[train_mask]

  # generate partitions
  ps, id_maps = kl_2partition(adj)
  for idx, (sub_adj, sub2fullid) in enumerate(zip(ps, id_maps)):
    if args.wrap_neighbor:
      sub_adj, sub2fullid = refine.wrap_neighbor(
        adj, sub_adj, sub2fullid, args.num_hop, train_nids=train_nids)
    # save to file
    pfile = '{}subadj_{}_{}hop.npz'.format(
      'wrap_' if args.wrap_neighbor else '',
      str(idx), args.num_hop
    )
    mapfile = '{}sub2fullid_{}_{}hop.npy'.format(
      'wrap_' if args.wrap_neighbor else '',
      str(idx), args.num_hop
    )
    pfile = os.path.join(partition_dataset, pfile)
    mapfile = os.path.join(partition_dataset, mapfile)
    spsp.save_npz(pfile, sub_adj)
    np.save(mapfile, sub2fullid)
  