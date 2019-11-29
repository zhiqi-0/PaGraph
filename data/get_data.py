
import numpy as np
import scipy.sparse


def get_graph_data(dataname):
  """
  Parames:
    dataname: shoud be a folder name, which contains
              adj.npz and feat.npy
  Returns:
    adj, feat, train_mask, val_mask, test_mask, labels
  """

  adj = scipy.sparse.load_npz(
    os.path.join(dataname, 'adj.npz')
  )
  feat = np.load(
    os.path.join(dataname, 'feat.npy')
  )
  return adj, feat


def get_struct(dataname):
  """
  Params:
    dataname: shoud be a folder name, which contains
              adj.npz in coo matrix format
  """
  adj = scipy.sparse.load_npz(
    os.path.join(dataname, 'adj.npz')
  )
  return adj


def get_masks(dataname):
  """
  Params:
    dataname: shoud be a folder name, which contains
              train_mask, val_mask, test_mask
  """
  train_mask = np.load(
    os.path.join(dataname, 'train.npy')
  )
  val_mask = np.load(
    os.path.join(dataname, 'val.npy')
  )
  test_mask = np.load(
    os.path.join(dataname, 'test.npy')
  )
  return train_mask, val_mask, test_mask


def get_labels(dataname):
  """
  Params:
    dataname: shoud be a folder name, which contains
              train_mask, val_mask, test_mask
  """
  labels = np.load(
    os.path.join(dataname, 'labels.npy')
  )
  return labels
