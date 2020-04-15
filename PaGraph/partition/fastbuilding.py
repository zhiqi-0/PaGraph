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
import torch.multiprocessing as mp


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
  
  full_eids = nf._edge_mapping
  subg = dgl_g.edge_subgraph(full_eids, preserve_nodes=False)
  sub2full = subg._parent_nid.tonumpy()
  tmax = np.max(sub2full)
  tmin = np.min(sub2full)
  train_nid = np.where(train_nid <= tmax, train_nid, tmin)
  subtrainid = subg.map_to_subgraph_nid(np.unique(train_nid))
  coo_adj = subg.adjacency_matrix_scipy(transpose=True, fmt='coo', return_edge_ids=False)
  vnum = len(sub2full)
  enum = coo_adj.row.shape[0]
  print('vertex#: {} edge#: {}'.format(vnum, enum))
  return coo_adj, sub2full, subtrainid


def worker_partition(rank, args, dgl_g, train_nid):
  labels = data.get_labels(args.dataset)
  partition_dataset = os.path.join(args.dataset, '{}naive'.format(args.partition))
  try:
    os.mkdir(partition_dataset)
  except FileExistsError:
    pass
  # get train vertices partitions
  chunk_size = int(len(train_nid) / args.partition)
  start_ofst = chunk_size * rank
  if rank == args.partition - 1:
    end_ofst = len(train_nid)
  else:
    end_ofst = start_ofst + chunk_size
  part_nid = train_nid[start_ofst:end_ofst]
  subadj, sub2fullid, subtrainid = get_sub_graph(dgl_g, part_nid, args.num_hops)
  sublabel = labels[sub2fullid[subtrainid]]
  # files
  subadj_file = os.path.join(
    partition_dataset,
    'subadj_{}.npz'.format(str(rank)))
  sub_trainid_file = os.path.join(
    partition_dataset,
    'sub_trainid_{}.npy'.format(str(rank)))
  sub_train2full_file = os.path.join(
    partition_dataset,
    'sub_train2fullid_{}.npy'.format(str(rank)))
  sub_label_file = os.path.join(
    partition_dataset,
    'sub_label_{}.npy'.format(str(rank)))
  spsp.save_npz(subadj_file, subadj)
  np.save(sub_trainid_file, subtrainid)
  np.save(sub_train2full_file, sub2fullid)
  np.save(sub_label_file, sublabel)


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
  np.random.shuffle(train_nid)

  mp.spawn(worker_partition, args=(args, dgl_g, train_nid), nprocs=args.partition, join=True)
  
  