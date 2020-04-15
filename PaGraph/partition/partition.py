import os
import argparse
import numpy as np
import scipy.sparse as spsp
import networkx as nx
from networkx.algorithms.community import kernighan_lin

import refine
from utils import *

def kl_2partition(coo_adj):
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
  return sub_coo_adjs, subid2fullid


def naive_partition(coo_adj, num, train_nids, num_hop=1):
  """
  Split the graph into node-equal partitions without considering
  cross edge num
  Return:
    sub_train_nids: sub graph train node ids under sub graph space
    sub_train_adjs: sub train adjs under coo matrix format
    sub_train2fullid: maps from sub graph id to full graph id
    store_nid_masks: masks for node id whose features stored in gpu under sub graph space
  """
  # Step 1: partition train_nids into num equaled size
  print("Generating Train Nodes Partitions....")
  np.random.shuffle(train_nids)
  train_node_num = len(train_nids)
  sub_train_node_num = int(train_node_num / num)
  sub_train_nids = []
  for subid in range(num):
    ofst_begin = subid * sub_train_node_num
    ofst_end = ofst_begin + sub_train_node_num if subid != num - 1 else train_node_num
    sub_train_nids.append(train_nids[ofst_begin:ofst_end])
  # Step 2: build train graph for each partition
  print("Generating Sub Training Graphs...")
  sub_train_adjs = []
  sub_train2fullid = []
  for subid in range(num):
    sub_adj, sub2fullid, valid_train_nids = refine.build_train_graph(
      coo_adj, sub_train_nids[subid], num_hop)
    sub_train_adjs.append(sub_adj)
    sub_train2fullid.append(sub2fullid)
    sub_train_nids[subid] = full2sub_nid(sub2fullid, valid_train_nids)
  # Step 3: get node ids with top-sub-train-node-num most out-neighbors
  print("Generating Cached Feature Policy...")
  def out_neighbor_num(nid, indptr):
    return indptr[nid + 1] - indptr[nid]
  out_vfunc = np.vectorize(out_neighbor_num, excluded=['indptr'])
  save_train_nid = False
  store_nid_masks = []
  for subid, (sub_adj, sub_train2full) in enumerate(zip(sub_train_adjs, sub_train2fullid)):
    sub_mask = np.zeros(sub_adj.shape[0], dtype=np.int)
    top_num = int(coo_adj.shape[0] / num)
    if save_train_nid:
      # ALREADY CONVERTED????
      nids = full2sub_nid(sub_train2full, sub_train_nids[subid])
      sub_mask[nids] = 1
      top_num -= len(nids)
    # calculate each node out neighbors
    sub_adj_csr = sub_adj.tocsr()
    out_degrees = out_vfunc(nid=np.arange(sub_adj.shape[0], dtype=np.int),
                            indptr=sub_adj_csr.indptr)
    nids = np.argsort(out_degrees)
    sub_mask[nids[-top_num:]] = 1
    store_nid_masks.append(sub_mask.astype(np.bool))
  return sub_train_nids, sub_train_adjs, sub_train2fullid, store_nid_masks


def kl(args):
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
      mask = np.load(train_mask_file).astype(np.bool)
      train_nids = np.arange(adj.shape[0])[mask]
    else:
      adj = spsp.load_npz(adj_file)
      feat = np.load(feat_file)
      mask = np.load(mask_file).astype(np.bool)
      label = np.load(label_file)
      nids = np.arange(adj.shape[0])[mask]
      print("Building Training Graphs...")
      adj, train2fullid = refine.build_train_graph(adj, nids, args.num_hop)
      train_feat = feat[train2fullid]
      train_label = label[train2fullid]
      train_mask = mask[train2fullid]
      train_nids = np.arange(adj.shape[0])[train_mask]
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
  isin_mask_vfunc = np.vectorize(include, excluded=['node_range'])
  print("Generating Partitions...")
  ps, id_maps = kl_2partition(adj)
  for idx, (sub_adj, sub2trainid) in enumerate(zip(ps, id_maps)):
    sub_adj_file = os.path.join(partition_dataset, 'subadj_naive_{}'.format(idx))
    sub2train_file = os.path.join(partition_dataset, 'sub2trainid_naive_{}'.format(idx))
    spsp.save_npz(sub_adj_file, sub_adj)
    np.save(sub2train_file, sub2trainid)

  for idx, (sub_adj, sub2trainid) in enumerate(zip(ps, id_maps)):
    # record train nid under full graph namespace before wrapping
    sub_train_nids_infull = sub2trainid[
      isin_mask_vfunc(nid=sub2trainid, node_range=train_nids)]
    # wrap num-hop neighbors
    if args.wrap_neighbor:
      print('Wrapping Partitions {}...'.format(idx))
      sub_adj, sub2trainid = refine.wrap_neighbor(
        adj, sub_adj, sub2trainid, args.num_hop, train_nids=train_nids)
    # convert train nid into sub graph namespace
    sub_train_nids = full2sub_nid(sub2trainid, sub_train_nids_infull)
    # save to file
    train_nid_file = 'train_{}_{}hop.npy'.format(str(idx), args.num_hop)
    pfile = '{}subadj_{}_{}hop.npz'.format(
      'wrap_' if args.wrap_neighbor else '',
      str(idx), args.num_hop
    )
    mapfile = '{}sub2trainid_{}_{}hop.npy'.format(
      'wrap_' if args.wrap_neighbor else '',
      str(idx), args.num_hop
    )
    train_nid_file = os.path.join(partition_dataset, train_nid_file)
    pfile = os.path.join(partition_dataset, pfile)
    mapfile = os.path.join(partition_dataset, mapfile)

    np.save(train_nid_file, sub_train_nids)
    spsp.save_npz(pfile, sub_adj)
    np.save(mapfile, sub2trainid)
  


def naive(args):
  # original dataset path
  adj_file = os.path.join(args.dataset, 'adj.npz')
  mask_file = os.path.join(args.dataset, 'train.npy')
  label_file = os.path.join(args.dataset, 'labels.npy')
  partition_dataset = os.path.join(args.dataset, '{}naive'.format(args.partition))
  try:
    os.mkdir(partition_dataset)
  except FileExistsError:
    pass
  full_adj = spsp.load_npz(adj_file)
  train_mask = np.load(mask_file).astype(np.bool)
  train_nids = np.arange(full_adj.shape[0], dtype=np.int)[train_mask]
  labels = np.load(label_file)
  sub_train_nids, sub_train_adjs, sub_train2fullid, store_nid_masks = \
    naive_partition(full_adj, args.partition, train_nids, args.num_hop)
  for subid in range(args.partition):
    train2full = sub_train2fullid[subid]
    sub_train_nids_infull = train2full[sub_train_nids[subid]]
    sub_label = labels[sub_train_nids_infull]
    subadj_file = os.path.join(
      partition_dataset,
      'subadj_{}.npz'.format(str(subid)))
    sub_trainid_file = os.path.join(
      partition_dataset,
      'sub_trainid_{}.npy'.format(str(subid)))
    sub_train2full_file = os.path.join(
      partition_dataset,
      'sub_train2fullid_{}.npy'.format(str(subid)))
    sub_storenid_file = os.path.join(
      partition_dataset,
      'sub_storenid_{}.npy'.format(str(subid)))
    sub_label_file = os.path.join(
      partition_dataset,
      'sub_label_{}.npy'.format(str(subid)))
    spsp.save_npz(subadj_file, sub_train_adjs[subid])
    np.save(sub_trainid_file, sub_train_nids[subid])
    np.save(sub_train2full_file, sub_train2fullid[subid])
    np.save(sub_storenid_file, store_nid_masks[subid])
    np.save(sub_label_file, sub_label)
  


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

  parser.add_argument("--kl", dest='kl', action='store_true')
  parser.set_defaults(kl=False)
  
  args = parser.parse_args()

  if args.kl:
    kl(args)
  else:
    naive(args)