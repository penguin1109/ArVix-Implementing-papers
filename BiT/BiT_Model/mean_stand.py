## weight standardization을 적용한 convolution layer
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

## implementation of Weight Standardization
# weight standardization을 한다는 것은 convolution filter의 평균을 0, 분산을 1로 만드는 것이다.
class StdConv2d(nn.Conv2d):
  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim = [1,2,3], keepdim = True, unbiased = False) ## variance(분산), mean(평균)
    w = (w-m)/torch.sqrt(v + 1e-10)
    
    return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
