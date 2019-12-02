import os
import sys
# set environment
module_name ='PaGraph'
modpath = os.path.abspath('.')
if module_name in modpath:
  idx = modpath.find(module_name)
  modpath = modpath[:idx]
sys.path.append(modpath)


from multiprocessing import Process
import argparse, time, math
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '16'
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import time

from PaGraph.model.mxnet.gcn_nssc import GCNSampling, GCNInfer

def main(args):
    g = dgl.contrib.graph_store.create_graph_from_store(args.graph_name, "shared_mem")
    # We need to set random seed here. Otherwise, all processes have the same mini-batches.
    mx.random.seed(g.worker_id)
    features = g.nodes[:].data['features']
    labels = g.nodes[:].data['labels']
    train_mask = g.nodes[:].data['train_mask']
    val_mask = g.nodes[:].data['val_mask']
    test_mask = g.nodes[:].data['test_mask']

    if args.num_gpus > 0:
        ctx = mx.gpu(g.worker_id % args.num_gpus)
    else:
        ctx = mx.cpu()

    train_nid = mx.nd.array(np.nonzero(train_mask.asnumpy())[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(test_mask.asnumpy())[0]).astype(np.int64)

    n_classes = len(np.unique(labels.asnumpy()))
    n_train_samples = train_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()

    n0_feats = g.nodes[0].data['features']
    in_feats = n0_feats.shape[1]
    g_ctx = n0_feats.context

    degs = g.in_degrees().astype('float32').as_in_context(g_ctx)
    norm = mx.nd.expand_dims(1./degs, 1)
    g.set_n_repr({'norm': norm})

    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        mx.nd.relu,
                        args.dropout,
                        prefix='GCN')

    model.initialize(ctx=ctx)
    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           args.n_layers,
                           mx.nd.relu,
                           prefix='GCN')

    infer_model.initialize(ctx=ctx)

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create('local'))

    # initialize graph
    dur = []
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
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)
            step += 1

            batch_dur.append(time.time() - batch_start_time)
            if g.worker_id == 0:
              print('epoch [{}] step [{}]. Batch average time(s): {:.4f}'
                .format(epoch + 1, step, np.mean(np.array(batch_dur))))

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.
        num_tests = 0

        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid):
            nf.copy_from_parent(ctx=ctx)
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1)
            batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
            num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()
            num_tests += nf.layer_size(-1)
            break

        print("Test Accuracy {:.4f}". format(num_acc/num_tests))
    
    print("parent ends")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--graph-name", type=str, default="",
            help="graph name")
    parser.add_argument("--num-feats", type=int, default=100,
            help="the number of features")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--num-gpus", type=int, default=0,
            help="the number of GPUs to train")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=3000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=2,
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

    print(args)

    main(args)