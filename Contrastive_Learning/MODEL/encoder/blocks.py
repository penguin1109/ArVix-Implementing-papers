""" Diverse Residual Blocks
- PostResBlock: 별건 아니고 모든 convolution layer들을 거친 후에 residual connection으로 input feature map을 그대로 local skip connection으로 더해 준다.
- Convolution, Normalization, Activation Layer의 순서를 달리 한다.
- Skip Connection의 addition의 순서를 달리한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


## Denoiser에서 GaussNet에 사용될 residual block.
# Gauss Net에서 사용하는 Gauss Block의 구조와 거의 유사하도록 하였다.
class PostResBlock(nn.Module):
  def __init__(self, 
               channel_in, 
               channel_out,
               stride,
               act_layer= nn.Tanh(), # nn.ReLU(inplace=True),
               conv_first=False,
               pre_norm=True,
               pre_act=True):
    super(PostResBlock, self).__init__()
    if stride != 1:
      self.downsample = nn.Sequential(
          nn.Conv2d(channel_in, channel_out, stride=stride, padding=0, kernel_size=1),
          nn.InstanceNorm2d(channel_out)
      )
    else:
      self.downsample = nn.Sequential(
          nn.Conv2d(channel_in, channel_out, stride=1, kernel_size=1, padding=0),
          nn.InstanceNorm2d(channel_out)
      )

    """ Args
    pre_norm: bool 
    pre_act: bool
    conv_first: bool 
      - True이면 conv-norm-relu
      - False이면 relu-conv-norm
    pre_norm = True, pre_act = False이면 Original Residual Block
    pre_norm = False, pre_act = False이면 Norm after Addition Residual Block
    pre_norm = True, pre_act = True이면 Activation before Addition Residual Block
    """
    self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=(1,1), bias=False, stride=1)
    self.conv2 = nn.Conv2d(channel_out,channel_out, kernel_size=(3,3), padding=(1,1), stride=stride, padding_mode="reflect", bias=False)
    self.conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=(1,1), padding=(0,0),stride=1, bias=False)

    self.norm1 = nn.InstanceNorm2d(channel_out, affine=True)
    self.norm2 = nn.InstanceNorm2d(channel_out, affine=True)
    self.norm3 = nn.InstanceNorm2d(channel_out, affine=True)

    self.relu = act_layer
    self.conv_first = conv_first
    if self.conv_first == False:
      self.pre_norm=True
      self.pre_act=True
    else:
      self.pre_norm = pre_norm
      self.pre_act = pre_act
  
  def forward(self, x):
    """ Explanation
    x: (B, C, H, W)
    기본적으로 relu-conv-norm의 순서이다. -> default로는 마지막에 addition 만 한다.
    """
    inp = x.clone()
    ## (1) PHASE 1
    # x = self.conv1(x)
    if self.conv_first:
      x = self.relu(self.norm1(self.conv1(x)))
    else:
      x = self.norm1(self.conv1(self.relu(x)))

    ## (2) PHASE 2
    if self.conv_first:
      x = self.relu(self.norm2(self.conv2(x)))
    else:
      x = self.norm2(self.conv2(self.relu(x)))

    ## (3) PHASE 3
    if self.conv_first == False:
      x = self.norm3(self.conv3(self.relu(x)))
    else:
      x = self.conv3(x)
      if self.pre_act:
        x = self.relu(x)
      if self.pre_norm:
        x = self.norm3(x)
    
    if self.downsample:
      inp = self.downsample(inp)
    out = inp + x
    if self.pre_norm == False:
      out = self.norm3(out)
    if self.pre_act == False:
      out = self.relu(out)
    
    return out
    




