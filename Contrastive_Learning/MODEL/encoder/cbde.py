import torch
import torch.nn as nn
import torch.nn.functional as F
from .simsiam import SimSiam
from .moco import MoCo
from .res_encoder import ResEncoder

class CBDE(nn.Module):
  def __init__(self, batch_size, moco = False):
    super(CBDE, self).__init__()
    self.dim = 256
    self.moco = moco
    self.batch_size = batch_size
    if self.moco == False:
      self.E = SimSiam(hidden_dim=128, pred_dim=256,)
    else:
      self.E = MoCo(ResEncoder, dim=self.dim, K = self.dim * batch_size)
  
  def forward(self, query_x, key_x, training=True):
    if self.moco:
      if training:
        feature, logits, labels, inter = self.E(query_x, key_x, training = True)
        return feature, logits, labels, inter
      else:
        feature, inter = self.E(query_x, key_x, training = False)
        return feature, inter
    else:
      if training:
        inter, query_feat, key_feat, query_cls, key_cls = self.E(query_x, key_x, training = True)
        return inter, query_feat, key_feat, query_cls, key_cls
      else:
        inter, query_cls = self.E(query_x, key_x, training = False)
        return inter, query_cls