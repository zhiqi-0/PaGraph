import os
import sys
# set environment
module_name ='PaGraph'
modpath = os.path.abspath('.')
if module_name in modpath:
  idx = modpath.find(module_name)
  modpath = modpath[:idx]
sys.path.append(modpath)
os.environ['OMP_NUM_THREADS'] = '16'

import argparse, time, math
import numpy as np
import os
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
import time

from PaGraph.model.mxnet.gcn_nssc import GCNSampling, GCNInfer
import PaGraph.data as data

def trainer(args):
  
  # load data
  dataname = os.path.basename(args.dataset)
  g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")
  labels = data.get_labels(args.dataset)
  n_classes = len(np.unique(labels))
  ctx = mx.gpu(g.worker_id % args.ngpu)
  # masks for semi-supervised learning
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nid = mx.nd.array(np.nonzero(train_mask)[0]).astype(np.int64)
  val_nid = mx.nd.array(np.nonzero(train_mask)[0]).astype(np.int64)
  test_nid = mx.nd.array(np.nonzero(test_mask)[0]).astype(np.int64)
  # to mxnet ndarrays
  labels = mx.nd.array(labels, ctx=ctx)
  train_mask = mx.nd.array(train_mask, ctx=ctx)
  val_mask = mx.nd.array(val_mask, ctx=ctx)
  test_mask = mx.nd.array(test_mask, ctx=ctx)

  # prepare model
  model = GCNSampling(args.feat_size,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      mx.nd.relu,
                      args.dropout,
                      prefix='GCN')

  model.initialize(ctx=ctx)
  loss_fcn = gluon.loss.SoftmaxCELoss()
  
  infer_model = GCNInfer(args.feat_size,
                         args.n_hidden,
                         n_classes,
                         args.n_layers,
                         mx.nd.relu,
                         prefix='GCN')
  infer_model.initialize(ctx=ctx)

  optimizer = gluon.Trainer(model.collect_params(), 'adam',
                          {'learning_rate': args.lr, 'wd': args.weight_decay},
                          kvstore=mx.kv.create('local'))
  
  # start training
  epoch_dur = []
  batch_dur = []
  for epoch in range(args.n_epochs):
    step = 0
    for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                   args.num_neighbors,
                                                   neighbor_type='in',
                                                   shuffle=True,
                                                   num_workers=16,
                                                   num_hops=args.n_layers+1,
                                                   seed_nodes=train_nid):
      batch_start_time = time.time()

      nf.copy_from_parent(ctx=ctx)
      with mx.autograd.record():
        pred = model(nf)
        batch_nids = nf.layer_parent_nid(-1).as_in_context(ctx)
        label = labels[batch_nids]
        loss = loss_fcn(pred, label)
        loss = loss.sum() / len(batch_nids)
      loss.backward()
      optimizer.step(batch_size=1)
      
      step += 1
      batch_dur.append(time.time() - batch_start_time)
      if g.worker_id == 0 and step % 20 == 0:
        print('epoch [{}] step [{}]. Batch average time(s): {:.4f}'
              .format(epoch + 1, step, np.mean(np.array(batch_dur))))
  
  print('Training Finishes')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')

  parser.add_argument("--ngpu", type=int, default=None,
                      help="num of gpus")
  parser.add_argument("--dataset", type=str, default=None,
                      help="path to the dataset folder")
  # model arch
  parser.add_argument("--feat-size", type=int, default=300,
                      help='input feature size')
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="dropout probability")
  parser.add_argument("--n-hidden", type=int, default=32,
                      help="number of hidden gcn units")
  parser.add_argument("--n-layers", type=int, default=1,
                      help="number of hidden gcn layers")
  # training hyper-params
  parser.add_argument("--lr", type=float, default=3e-2,
                      help="learning rate")
  parser.add_argument("--n-epochs", type=int, default=60,
                      help="number of training epochs")
  parser.add_argument("--batch-size", type=int, default=2500,
                      help="batch size")
  parser.add_argument("--weight-decay", type=float, default=0,
                      help="Weight for L2 loss")
  # sampling hyper-params
  parser.add_argument("--num-neighbors", type=int, default=2,
                      help="number of neighbors to be sampled")
  
  args = parser.parse_args()

  trainer(args)