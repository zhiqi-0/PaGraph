import os
import sys
# set environment
module_name ='PaGraph'
modpath = os.path.abspath('.')
if module_name in modpath:
  idx = modpath.find(module_name)
  modpath = modpath[:idx]
sys.path.append(modpath)

import numpy as np
import numba
import torch
from dgl import DGLGraph
import dgl.utils
from .utils import pinclude

class GraphCacheServer:
  """
  Manage graph features
  Automatically fetch the feature tensor from CPU or GPU
  """
  def __init__(self, graph, node_num, nid_map, gpuid):
    """
    Paramters:
      graph:   should be created from `dgl.contrib.graph_store`
      node_num: should be sub graph node num
      nid_map: numpy array. map from local node id to full graph id.
               used in fetch features from remote
    """
    self.graph = graph
    self.node_num = node_num
    self.nid_map = nid_map
    
    # masks for manage the feature locations
    self.cpu_flag = np.ones(self.node_num, dtype=np.bool)
    self.gpu_flag = np.zeros(self.node_num, dtype=np.bool)

    # gpu tensor cache
    self.gpuid = gpuid
    self.cached_num = 0
    self.dims = {}          # {'field name': dims of the tensor data of a node}
    self.gpu_fix_cache = dict() # {'field name': tensor data for cached nodes in gpu}
    with torch.cuda.device(self.gpuid):
      self.localid2cacheid = torch.cuda.LongTensor(node_num).fill_(0)
  
  
  def cache_fix_data(nids, data):
    """
    User should make sure tensor data under every field name should
    have same num (axis 0)
    Params:
      data: dict: {'field name': tensor data}
    """
    with torch.cuda.device(self.gpuid):
      for name in data:
        data_num = data[name].size(0)
        self.dims[name] = data[name].size(1)
        gpu_fix_cache[name] = data[name].cuda()
        self.localid2cacheid[nids] = torch.arange(data_num).cuda()
        if self.cached_num == 0:
          self.cached_num = data[name].size(0)
        elif self.cached_num != data[name].size(0):
          print("Error: tensor data under each field should have same num!")
          sys.exit(-1)
      

  
  def fetch_data(self, nodeflow):
    """
    copy feature from local GPU memory or
    remote CPU memory, which depends on feature
    current location.
    --Note: Should be paralleled
    Params:
      nodeflow: DGL nodeflow. all nids in nodeflow should
                under sub-graph space
    """
    for i in range(nodeflow.num_layers):
      sub_nid = nodeflow.layer_parent_nid(i)
      #nodeflow._node_frames[i] = FrameRef(Frame(num_rows=len(sub_nid)))
      # get mask
      sub_nid_gpu_mask = self.gpu_flag[sub_nid]
      sub_nid_in_gpu = sub_nid[sub_nid_gpu_mask].cuda(self.gpuid)
      sub_nid_cpu_mask = self.cpu_flag[sub_nid]
      sub_nid_in_cpu = sub_nid[sub_nid_cpu_mask]
      # create frame
      with torch.cuda.device(self.gpuid):
        frame = {key: torch.cuda.FloatTensor(len(sub_nid), self.dims[key]).fill_(0.) \
                  for key in self.gpu_fix_cache}
      ##NOTE: can be paralleled io
      # for gpu cached tensors: ##NOTE: Make sure it is in-place update!
      for key in self.gpu_fix_cache:
        cacheid = self.localid2cacheid[sub_nid_in_gpu]
        frame[name][sub_nid_in_gpu] = self.gpu_fix_cache[key][cacheid]
      # for cpu cached tensors: ##NOTE: Make sure it is in-place update!
      cpu_nid_infull = self.nid_map[sub_nid_in_cpu]
      cpu_data = self.graph._node_frames[cpu_nid_infull]
      with torch.cuda.device(self.gpuid):
        for key in self.gpu_fix_cache:
          frame[name][sub_nid_in_cpu] = cpu_data[name].cuda()
      nodeflow._node_frames[i] = FrameRef(frame)
