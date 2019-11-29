import sys
import os
# set environment
module_name ='PaGraph'
modpath = os.path.abspath('.')
if module_name in modpath:
  idx = modpath.find(module_name)
  modpath = modpath[:idx]
sys.path.append(modpath)

import argparse
import numpy as np
import torch
import dgl
from dgl import DGLGraph

import PaGraph.data as data

def main(args):
  coo_adj, feat = data.get_graph_data(args.dataset)

  graph_name = args.dataset
  graph = dgl.DGLGraph(coo_adj, readonly=True)
  features = torch.FloatTensor(feat)

  vnum = graph.number_of_nodes()
  enum = graph.number_of_edges()
  feat_size = feat.shape[1]

  print('=' * 30)
  print("Nodes Num: {}\tEdges Num: {}\nFeature Size: {}"
        .format(vnum, enum, feat_size)
  )

  # create server
  g = dgl.contrib.graph_store.create_graph_store_server(
        data.graph, graph_name,
        'shared_mem', args.num_workers, 
        False, edge_dir='in')
  
  # calculate norm for gcn
  dgl_g = DGLGraph(graph, readonly=True)
  norm = 1. / dgl_g.in_degrees().float().unsqueeze(1)
  del dgl_g

  # setup features and norms
  g.ndata['norm'] = norm
  g.ndata['features'] = features

  print('start running graph server on dataset: {}'.format(graph_name))
  g.run()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GraphServer')
  register_data_args(parser)

  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset folder path")
  
  parser.add_argument("--num-workers", type=int, default=1,
                      help="the number of workers")
  
  args = parser.parse_args()
  main(args)