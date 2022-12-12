import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math

class dcn_layer(nn.Module):
  def __init__(self, channel_in, channel_out,
               kernel_size=(3,3),
              offset_groups=1,
              bias=False):
    super(dcn_layer, self).__init__()
    self.offset_groups = offset_groups ## deformable group의 수
    self.channel_in = channel_in
    self.ksize = kernel_size
    self.padding_size = tuple(map(lambda x: (x - 1)//2, self.ksize))

    self.weight = nn.Parameter(
        torch.Tensor(channel_out, channel_in // self.offset_groups, kernel_size[0], kernel_size[1])
    )
    self.weight.data.zero_()
    if bias:
      self.bias = nn.Parameter(torch.Tensor(channel_out))
      self.bias.zero_()
    else:
      self.bias = None

    self.offset_mask = nn.Conv2d(
        channel_in * 2,
        self.offset_groups * 3 * kernel_size[0] * kernel_size[1],
        kernel_size = kernel_size,
        stride = (1,1),
        padding = self.padding_size,
        bias=True)
    self.reset_parameters()

  def reset_parameters(self):
    n = self.channel_in
    for k in self.ksize:
      n *= k
    stdv = 1. / math.sqrt(n)
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.zero_()

  def forward(self, input_feat, degrad_inter):
    """ Args
    input_feat: (B, C, H, W) convolution block 중간에 출력으로 나오는 feature map
    degrad_inter: (B, C, H, W) CBDE의 첫번째 layer의 output
    """
    concat=torch.cat([input_feat, degrad_inter], dim=1)
    out = self.offset_mask(concat)
    off_1, off_2, mask = torch.chunk(out, chunks = 3, dim = 1)
    offset = torch.cat([off_1, off_2], dim = 1)
    mask = torch.sigmoid(mask)
    
    out = deform_conv2d(
        input = input_feat.contiguous(), offset = offset,
        mask = mask, weight = self.weight,
        bias = self.bias, stride = 1, padding = 1
    )

    return out




class SFTLayer(nn.Module):
  def __init__(self, channel_in, channel_out):
    super(SFTLayer, self).__init__()
    self.gamma = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
        nn.ReLU(),
        nn.Conv2d(channel_out, channel_out, kernel_size=1, padding=0, bias=False)
    )
    self.beta = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
        nn.ReLU(),
        nn.Conv2d(channel_out, channel_out, kernel_size=1, padding=0, bias=False)
    )
  def forward(self, x, repr):
    """ Args
    x: input feature map (B, C, H, W)
    repr: degradation representation feature map (B, C, H, W)
    """
    gamma = self.gamma(repr)
    beta = self.beta(repr)

    transformed = (x*gamma) + beta
    return transformed

