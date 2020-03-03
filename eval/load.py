import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import argparse
import os

vnum = 4036538
batch_num = 438
batch_vnum = 37225

def init_process(rank, world_size, backend):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29501'
  dist.init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(rank)
  print('rank [{}] process successfully launches'.format(rank))

# batch num of livejournal: 438
# batch vertex# of livejournal (2 neighbor): 37225
# avg msg size of livejournal: 85.20MB bandwidth: 2.03GB/s
def loading(rank, word_size, args, backend='nccl'):
  init_process(rank, word_size, backend)
  print('generating data...')
  data = torch.rand((vnum, args.feat_size))
  print('start loading...')
  with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for epoch in range(args.n_epochs):
      for _ in range(batch_num):
        # random id
        ids = np.random.randint(vnum, size=(batch_vnum,), dtype=np.int64)
        tids = torch.tensor(ids)
        with torch.autograd.profiler.record_function('load'):
          fdata = data[tids].cuda()
        del fdata

  print(prof.key_averages().table(sort_by='cuda_time_total'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Load')

  parser.add_argument("--gpu", type=str, default='cpu',
                      help="gpu ids. such as 0 or 0,1,2")
  parser.add_argument("--feat-size", type=int, default=600,
                      help='input feature size')
  parser.add_argument("--n-epochs", type=int, default=10,
                      help="number of training epochs")
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpu_num = len(args.gpu.split(','))

  mp.spawn(loading, args=(gpu_num, args), nprocs=gpu_num, join=True)
