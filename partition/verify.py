import os
import argparse
import numpy as np
import scipy.sparse as spsp
import networkx as nx

from utils import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PartitionVerify')
  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")

  train_dataset = os.path.join(dataset, 'train')
  partition_dataset = os.path.join(dataset, 'partition')
  # full graph file
  full_adj_file = os.path.join(dataset, 'adj.npz')
  full_train_mask = os.path.join(dataset, 'train.npy')
  # train graph file
  train_adj_file = os.path.join(train_dataset, 'adj.npz')
  train2fullid_file = os.path.join(train_dataset, 'train2fullid.npy')
  # partition graph file
  pfile = ['wrap_subadj_{}_1hop.npz'.format(str(idx)) for idx in range(2)]
  pfile = [os.path.join(partition_dataset, p) for p in pfile]
  mapfile = ['wrap_sub2trainid_{}_1hop.npy'.format(str(idx)) for idx in range(2)]
  mapfile = [os.path.join(partition_dataset, mp) for mp in mapfile]

  full_adj = spsp.load_npz(full_adj_file)
  train_nid_full = np.arange(full_adj.shape[0])[np.load(full_train_mask).astype(np.bool)]

  train_adj = spsp.load_npz(train_adj_file)
  train2fullid = np.load(train2fullid_file)

  sub_adjs = [spsp.load_npz(subadj_file) for subadj_file in pfile]
  sub2trainids = [train2fullid[np.load(submap_file)] for submap_file in mapfile]

  # draw full graph
  print('train nids:', train_nid_full)
  draw_graph(full_adj)

  # draw train graph
  draw_graph(train_adj, train2fullid)

  # draw sub graph
  draw_graph(sub_adjs[0], train2fullid[sub2trainids[0]], pos=nx.spring_layout)
  draw_graph(sub_adjs[1], train2fullid[sub2trainids[1]], pos=nx.spring_layout)

