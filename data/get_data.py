
import numpy as np
from dgl.data import load_data

def load_adj_feature(adj_npy, feat_npy):
  """
  Load adjacancy matrix from adj_npy file.
  Load feature matrix from feat_npy file.
  """
  adj = np.load(adj_npy)
  feat = np.load(feat_npy)
  return adj, feat


def get_data(dataname):
  """
  Parames:
    dataname: shoud be a folder name, which contains
              adj.npy and feat.npy
  Returns:
    adj, feat
  """
  adj, feat = load_adj_feature('adj.npy', 'feat.npy')
  return adj, feat
  