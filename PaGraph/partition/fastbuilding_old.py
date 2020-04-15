import os
import sys
# set environment
module_name ='PaGraph'
modpath = os.path.abspath('.')
if module_name in modpath:
  idx = modpath.find(module_name)
  modpath = modpath[:idx]
sys.path.append(modpath)

import dgl
from dgl import DGLGraph
import torch
import numpy as np
import scipy.sparse as spsp
import argparse
import PaGraph.data as data


def get_sub_graph(dgl_g, train_nid, num_hops):
  nfs = []
  for nf in dgl.contrib.sampling.NeighborSampler(dgl_g, len(train_nid),
                                                 dgl_g.number_of_nodes(),
                                                 neighbor_type='in',
                                                 shuffle=False,
                                                 num_workers=16,
                                                 num_hops=num_hops,
                                                 seed_nodes=train_nid,
                                                 prefetch=False):
    nfs.append(nf)
  
  assert(len(nfs) == 1)
  nf = nfs[0]
  #full_eids = nf._edge_mapping
  #subg = dgl_g.edge_subgraph(full_eids, preserve_nodes=False)
  #sub2full = subg._parent_nid.tonumpy()
  #tmax = np.max(sub2full)
  #tmin = np.min(sub2full)
  #train_nid = np.where(train_nid <= tmax, train_nid, tmin)
  #subtrainid = subg.map_to_subgraph_nid(np.unique(train_nid))
  #coo_adj = subg.adjacency_matrix_scipy(transpose=True, fmt='coo', return_edge_ids=False)
  
  full_edge_src = []
  full_edge_dst = []
  for i in range(nf.num_blocks):
    nf_src_nids, nf_dst_nids, _ = nf.block_edges(i, remap_local=False)
    full_edge_src.append(nf.map_to_parent_nid(nf_src_nids))
    full_edge_dst.append(nf.map_to_parent_nid(nf_dst_nids))
  full_srcs = torch.cat(tuple(full_edge_src)).numpy()
  full_dsts = torch.cat(tuple(full_edge_dst)).numpy()
  # set up mappings
  sub2full = np.unique(np.concatenate((full_srcs, full_dsts)))
  full2sub = np.zeros(np.max(sub2full) + 1, dtype=np.int64)
  full2sub[sub2full] = np.arange(len(sub2full), dtype=np.int64)
  # map to sub graph space
  sub_srcs = full2sub[full_srcs]
  sub_dsts = full2sub[full_dsts]
  vnum = len(sub2full)
  enum = len(sub_srcs)
  print('vertex#: {} edge#: {}'.format(vnum, enum))
  data = np.ones(sub_srcs.shape[0], dtype=np.uint8)
  coo_adj = spsp.coo_matrix((data, (sub_srcs, sub_dsts)), shape=(vnum, vnum))
  # train nid
  tnid = nf.layer_parent_nid(-1).numpy()
  valid_t_max = np.max(sub2full)
  valid_t_min = np.min(tnid)
  tnid = np.where(tnid <= valid_t_max, tnid, valid_t_min)
  subtrainid = full2sub[np.unique(tnid)]
  return coo_adj, sub2full, subtrainid


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='FastBuilding')
  parser.add_argument("--dataset", type=str, default=None,
                      help="path to the dataset folder")
  parser.add_argument("--num-hops", type=int, default=1,
                      help="num hops for the extended graph")
  parser.add_argument("--partition", type=int, default=2,
                      help="partition number")
  args = parser.parse_args()

  # load data
  adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz'))
  dgl_g = DGLGraph(adj, readonly=True)
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nid = np.nonzero(train_mask)[0].astype(np.int64)
  # shuffle
  np.random.shuffle(train_nid)
  labels = data.get_labels(args.dataset)

  # save
  adj_file = os.path.join(args.dataset, 'adj.npz')
  mask_file = os.path.join(args.dataset, 'train.npy')
  label_file = os.path.join(args.dataset, 'labels.npy')
  partition_dataset = os.path.join(args.dataset, '{}naive'.format(args.partition))
  try:
    os.mkdir(partition_dataset)
  except FileExistsError:
    pass

  chunk_size = int(len(train_nid) / args.partition)
  for pid in range(args.partition):
    start_ofst = chunk_size * pid
    if pid == args.partition - 1:
      end_ofst = len(train_nid)
    else:
      end_ofst = start_ofst + chunk_size
    part_nid = train_nid[start_ofst:end_ofst]
    subadj, sub2fullid, subtrainid = get_sub_graph(dgl_g, part_nid, args.num_hops)
    sublabel = labels[sub2fullid[subtrainid]]
    # files
    subadj_file = os.path.join(
      partition_dataset,
      'subadj_{}.npz'.format(str(pid)))
    sub_trainid_file = os.path.join(
      partition_dataset,
      'sub_trainid_{}.npy'.format(str(pid)))
    sub_train2full_file = os.path.join(
      partition_dataset,
      'sub_train2fullid_{}.npy'.format(str(pid)))
    sub_label_file = os.path.join(
      partition_dataset,
      'sub_label_{}.npy'.format(str(pid)))
    spsp.save_npz(subadj_file, subadj)
    np.save(sub_trainid_file, subtrainid)
    np.save(sub_train2full_file, sub2fullid)
    np.save(sub_label_file, sublabel)
  
  
