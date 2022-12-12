""" DGRN: Degradation Guided Reconstruction Layer
DGM: Degradation Guided Module
DGB: Degradation Guided Block
DGG: Degradation Guided Group
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SFTLayer, dcn_layer

class DGM(nn.Module):
  def __init__(self, channel_in, channel_out, kernel_size=3):
    super(DGM, self).__init__()
    self.sft = SFTLayer(channel_in, channel_out)
    self.dcn = dcn_layer(channel_in, channel_out, kernel_size = (kernel_size, kernel_size))

  def forward(self, input_feat, degrad_repr):
    sft = self.sft(input_feat, degrad_repr)
    dcn = self.dcn(input_feat, degrad_repr)
    out = sft + dcn

    return out + input_feat

  
class DGB(nn.Module):
  def __init__(self, feat, kernel_size = 3):
    super(DGB, self).__init__()
    self.dgm1 = DGM(feat, feat)
    self.dgm2 = DGM(feat, feat)
    self.conv1 = nn.Conv2d(feat, feat, kernel_size = kernel_size, stride=1, padding = (kernel_size-1)//2)
    self.conv2 = nn.Conv2d(feat, feat, kernel_size = kernel_size, stride=1, padding = (kernel_size-1)//2)

    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, input_feat, degrad_repr):
    out = self.relu(self.dgm1(input_feat, degrad_repr))
    out = self.conv1(out)
    out = self.relu(self.dgm2(out, degrad_repr))
    out = self.conv2(out)

    return out + input_feat

class DGG(nn.Module):
  def __init__(self, channel_in, feat,
               block_n=5):
    super(DGG, self).__init__()
    self.body = nn.ModuleList([DGB(feat) for _ in range(block_n)])
    self.final_conv = nn.Conv2d(feat, feat, kernel_size=3, padding=1, stride=1)

  def forward(self, input_feat, degrad_repr):
    """ Args
    input_feat: (B, C, H, W) Feature Map
    degrad_repr: (B, C, H, W) Representation Map of the degradation level
    """
    x = input_feat.clone()
    for idx, layer in enumerate(self.body):
      input_feat = layer(input_feat, degrad_repr)
    input_feat = self.final_conv(input_feat)

    return x + input_feat

class DGRN(nn.Module):
  def __init__(self, channel_in, channel_out, feat,
               group_n=5):
    super(DGRN, self).__init__()
    self.head = nn.Conv2d(channel_in, feat, kernel_size=3, stride=1, padding=1)
    self.groups = nn.ModuleList([
        DGG(feat, feat) for _ in range(group_n)
    ])
    self.tail = nn.Conv2d(feat, channel_out, kernel_size=3, stride=1, padding=1)
  
  def forward(self, x, degradation):
    head = self.head(x)
    out = head.clone()
    for idx, layer in enumerate(self.groups):
      out = layer(out, degradation)
    out = out + head
    out = self.tail(out)

    return out


      
