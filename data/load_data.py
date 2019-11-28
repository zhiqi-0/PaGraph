
import numpy as np

def load_adj_feature(adj_npy, feat_npy):
  """
  Load adjacancy matrix from adj_npy file.
  Load feature matrix from feat_npy file.
  """
  adj = np.load(adj_npy)
  feat = np.load(feat_npy)
  return adj, feat
  