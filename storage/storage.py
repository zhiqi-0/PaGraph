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
from dgl.frame import Frame, FrameRef
import dgl.utils

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
      nid_map: torch tensor. map from local node id to full graph id.
               used in fetch features from remote
    """
    self.graph = graph
    self.node_num = node_num
    self.nid_map = nid_map
    
    # masks for manage the feature locations: default in CPU
    self.cpu_flag = torch.ones(self.node_num).bool()
    self.gpu_flag = torch.zeros(self.node_num).bool()

    # gpu tensor cache
    self.gpuid = gpuid
    self.cached_num = 0
    self.full_cached = False
    self.dims = {}          # {'field name': dims of the tensor data of a node}
    self.gpu_fix_cache = dict() # {'field name': tensor data for cached nodes in gpu}
    with torch.cuda.device(self.gpuid):
      self.localid2cacheid = torch.cuda.LongTensor(node_num).fill_(0)
  

  def get_feat_from_server(self, nids, embed_names, to_gpu=False):
    """
    Fetch features of `nids` from remote server in shared CPU
    Params
      g: created from `dgl.contrib.graph_store.create_graph_from_store`
      nids: required node ids in local graph
      embed_names: field name list, e.g. ['features', 'norm']
    Return:
      feature tensors of these nids (in CPU)
    """
    nids_in_full = self.nid_map[nids]
    cpu_frame = self.graph._node_frame[dgl.utils.toindex(nids_in_full)]
    data_frame = {}
    for name in embed_names:
      if to_gpu:
        data_frame[name] = cpu_frame[name].cuda(self.gpuid)
      else:
        data_frame[name] = cpu_frame[name]
    return data_frame
  
  
  def cache_fix_data(self, nids, data, is_full=False):
    """
    User should make sure tensor data under every field name should
    have same num (axis 0)
    Params:
      nids: numpy arrary: node ids to be cached in local graph.
            should be equal to data rows
      data: dict: {'field name': tensor data}
    """
    rows = nids.shape[0]
    tnids = torch.tensor(nids).cuda(self.gpuid)
    self.localid2cacheid[tnids] = torch.arange(rows).cuda(self.gpuid)
    self.cached_num = rows
    for name in data:
      data_rows = data[name].size(0)
      assert (rows == data_rows)
      self.dims[name] = data[name].size(1)
      self.gpu_fix_cache[name] = data[name].cuda(self.gpuid)
    # setup flags
    self.cpu_flag[tnids] = False
    self.gpu_flag[tnids] = True
    self.full_cached = is_full

  
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
    if self.full_cached:
      self.fetch_from_cache(nodeflow)
      return
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
        frame = {name: torch.cuda.FloatTensor(len(sub_nid), self.dims[name]) \
                  for name in self.gpu_fix_cache}
      ##NOTE: can be paralleled io
      # for gpu cached tensors: ##NOTE: Make sure it is in-place update!
      for name in self.gpu_fix_cache:
        cacheid = self.localid2cacheid[sub_nid_in_gpu]
        frame[name][sub_nid_in_gpu] = self.gpu_fix_cache[name][cacheid]
      # for cpu cached tensors: ##NOTE: Make sure it is in-place update!
      cpu_nid_infull = self.nid_map[sub_nid_in_cpu]
      cpu_data = self.graph._node_frame[dgl.utils.toindex(cpu_nid_infull)]
      for name in self.gpu_fix_cache:
        frame[name][sub_nid_in_cpu] = cpu_data[name].cuda(self.gpuid)
      nodeflow._node_frames[i] = FrameRef(frame)


  def fetch_from_cache(self, nodeflow):
    for i in range(nodeflow.num_layers):
      #nid = dgl.utils.toindex(nodeflow.layer_parent_nid(i))
      tnid = nodeflow.layer_parent_nid(i).cuda(self.gpuid)
      frame = {}
      for name in self.gpu_fix_cache:
        frame[name] = self.gpu_fix_cache[name][tnid]
      nodeflow._node_frames[i] = FrameRef(Frame(frame))
