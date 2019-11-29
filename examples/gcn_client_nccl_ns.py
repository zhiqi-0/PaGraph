import os
import sys
# set environment
module_name ='PaGraph'
modpath = os.path.abspath('.')
if module_name in modpath:
  idx = modpath.find(module_name)
  modpath = modpath[:idx]
sys.path.append(modpath)

import argparse, time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import dgl

from PaGraph.model.gcn_ns import GCNSampling, GCNInfer

def init_process(rank, world_size, backend):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29501'
  dist.init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(rank)
  print('rank [{}] process successfully launches'.format(rank))

def trainer(rank, world_size, args, backend='nccl'):
  # init multi process
  init_process(rank, world_size, backend)
  # load data
  g = dgl.contrib.graph_store.create_graph_from_store(args.dataset, "shared_mem")
  features = g.nodes[:].data['features']
  norm = g.nodes[:].data['norm']
  labels = g.nodes[:].data['labels']
  train_mask = g.nodes[:].data['train_mask']
  val_mask = g.nodes[:].data['val_mask']
  test_mask = g.nodes[:].data['test_mask']
  in_feats = features.shape[1]
  n_classes = len(np.unique(labels.numpy()))

  n_train_samples = train_mask.sum().item()
  n_val_samples = val_mask.sum().item()
  n_test_samples = test_mask.sum().item()

  train_nid = np.nonzero(train_mask.numpy())[0].astype(np.int64)
  test_nid = np.nonzero(test_mask.numpy())[0].astype(np.int64)

  # prepare model
  model = GCNSampling(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout)
  loss_fcn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
  model.cuda(rank)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  ctx = torch.device(rank)

  # start training
  epoch_dur = []
  batch_dur = []
  for epoch in range(args.n_epochs):
    model.train()
    epoch_start_time = time.time()
    step = 0
    for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                  args.num_neighbors,
                                                  neighbor_type='in',
                                                  shuffle=True,
                                                  num_workers=8,
                                                  num_hops=args.n_layers+1,
                                                  seed_nodes=train_nid,
                                                  prefetch=True):
      batch_start_time = time.time()
      nf.copy_from_parent(ctx=ctx)
      batch_nids = nf.layer_parent_nid(-1)
      label = g.nodes[batch_nids].data['labels']
      label = label.cuda(rank, non_blocking=True)
      
      pred = model(nf)
      loss = loss_fcn(pred, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      step += 1
      batch_dur.append(time.time() - batch_start_time)
      if rank == 0 and step % 1 == 0:
        print('epoch [{}] step [{}]. Batch average time(s): {:.4f}'
              .format(epoch + 1, step, np.mean(np.array(batch_dur))))
    if rank == 0:
      epoch_dur.append(time.time() - epoch_start_time)
      print('Epoch average time: {:.4f}'.format(np.mean(np.array(epoch_dur[2:]))))
  
  if rank == 0:
    infer_model = GCNInfer(in_feats,
                   args.n_hidden,
                   n_classes,
                   args.n_layers,
                   F.relu)
    infer_model.cuda(ctx)
    for infer_param, param in zip(infer_model.parameters(), model.module.parameters()):    
      infer_param.data.copy_(param.data)
    num_acc = 0.
    for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                         g.number_of_nodes(),
                                                         neighbor_type='in',
                                                         num_workers=32,
                                                         num_hops=args.n_layers+1,
                                                         seed_nodes=test_nid):
      nf.copy_from_parent(ctx=ctx)
      infer_model.eval()
      with torch.no_grad():
        pred = infer_model(nf)
        batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
        batch_labels = labels[batch_nids].to(ctx)
        num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

    print("Test Accuracy {:.4f}".format(num_acc / n_test_samples))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')
  register_data_args(parser)
  parser.add_argument("--dropout", type=float, default=0.5,
          help="dropout probability")
  parser.add_argument("--gpu", type=str, default='cpu',
          help="gpu ids. such as 0 or 0,1,2")
  parser.add_argument("--lr", type=float, default=3e-2,
          help="learning rate")
  parser.add_argument("--n-epochs", type=int, default=200,
          help="number of training epochs")
  parser.add_argument("--batch-size", type=int, default=1000,
          help="batch size")
  parser.add_argument("--num-neighbors", type=int, default=3,
          help="number of neighbors to be sampled")
  parser.add_argument("--n-hidden", type=int, default=16,
          help="number of hidden gcn units")
  parser.add_argument("--n-layers", type=int, default=1,
          help="number of hidden gcn layers")
  parser.add_argument("--self-loop", action='store_true',
          help="graph self-loop (default=False)")
  parser.add_argument("--weight-decay", type=float, default=5e-4,
          help="Weight for L2 loss")
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpu_num = len(args.gpu.split(','))

  mp.spawn(trainer, args=(gpu_num, args), nprocs=gpu_num, join=True)
  