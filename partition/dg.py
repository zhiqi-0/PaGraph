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
import random


def in_neighbors(csc_adj, nid):
  return csc_adj.indices[csc_adj.indptr[nid]: csc_adj.indptr[nid+1]]


def dg_max_score(score, p_vnum):
  ids = np.argsort(score)[-2:]
  if score[ids[0]] != score[ids[1]]:
    return ids[1]
  else:
    return ids[0] if p_vnum[ids[0]] < p_vnum[ids[1]] else ids[1]


def dg_ind(adj, neighbors, belongs, p_vnum, r_vnum, pnum):
  """
  Params:
    neighbor: in-neighbor vertex set
    belongs: np array, each vertex belongings to which partition
    p_vnum: np array, each partition total vertex w/o. redundancy
    r_vnum: np array, each partition total vertex w/. redundancy
    pnum: partition number
  """
  com_neighbor = np.ones(pnum, dtype=np.int64)
  score = np.zeros(pnum, dtype=np.float32)
  # count belonged vertex
  neighbor_belong = belongs[neighbors]
  belonged = neighbor_belong[np.where(neighbor_belong != -1)]
  pid, freq = np.unique(belonged, return_counts=True)
  com_neighbor[pid] += freq
  avg_num = adj.shape[0] * 0.65 / pnum # need modify to match the train vertex num
  score = com_neighbor * (-p_vnum + avg_num) / (r_vnum + 1)
  return score


def dg(args, adj, train_nids):
  csc_adj = adj.tocsc()
  vnum = adj.shape[0]
  belongs = -np.ones(vnum, dtype=np.int8)
  r_belongs = [-np.ones(vnum, dtype=np.int8) for _ in range(args.partition)]
  p_vnum = np.zeros(args.partition, dtype=np.int64)
  r_vnum = np.zeros(args.partition, dtype=np.int64)

  progress = 0
  #for nid in range(0, train_nids):
  print(train_nids.shape[0])
  for nid in train_nids:  
    neighbors = in_neighbors(csc_adj, nid)
    score = dg_ind(csc_adj, neighbors, belongs, p_vnum, r_vnum, args.partition)
    ind = dg_max_score(score, p_vnum)
    if belongs[nid] == -1:
      belongs[nid] = ind
      p_vnum[ind] += 1
      #random_num = random.random()
      #if random_num < 0.65:
      neighbors = np.append(neighbors, nid)
      for neigh_nid in neighbors:
        if r_belongs[ind][neigh_nid] == -1:
          r_belongs[ind][neigh_nid] = 1
          r_vnum[ind] += 1
    # progress
    if int(vnum * progress / 100) <= nid:
      sys.stdout.write(' {}%\r'.format(progress))
      sys.stdout.flush()
      progress += 1
  
  for pid in range(args.partition):
    p_trainids = np.where(belongs == pid)[0]
    np.save('{}trainids'.format(pid), p_trainids)
    print('redundancy vertex#  ', r_vnum[pid])
    print('original vertex# ', p_vnum[pid])
    #print('orginal vertex: ', np.where(belongs == pid)[0])
    #print('redundancy vertex: ', np.where(r_belongs[pid] != -1)[0])

  


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Partition')
  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")
  parser.add_argument("--partition", type=int, default=2,
                      help="num of partitions")
  parser.add_argument("--num-hop", type=int, default=1,
                      help="num of hop neighbors required for a batch")
  args = parser.parse_args()

  adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz'))
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nids = np.nonzero(train_mask)[0].astype(np.int64)
  dg(args, adj, train_nids)