import torch
import torch.nn as nn

class ResNet(nn.Module):
  def __init__(self, n = 7, res_option = 'A', use_dropout = False):
    super(ResNet, self).__init__()
    self.res_option = res_option
    self.use_dropout = use_dropout
    ## Head
    self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
    self.norm1 = nn.BatchNorm2d(16)
    self.relu1 = nn.ReLU(inplace = True)

    ## Res Unit
    self.res_layer1 = self._make_layer(n, 16, 16, 1)
    self.res_layer2 = self._make_layer(n, 32, 16, 2)
    self.res_layer3 = self._make_layer(n, 64, 32, 2)
    self.avgpool = nn.AvgPool2d(8)
    self.linear = nn.Linear(64, 10)
  
  def _make_layer(self, layer_count, channels, channels_in, stride):
    ## channel의 수를 2배로 늘리는 
    return nn.Sequential(
        ResBlock(channels, channels_in, stride, res_option = self.res_option,  use_dropout = self.use_dropout), 
        *[ResBlock(channels) for _ in range(layer_count-1)]
    )
  
  def forward(self, x):
    out = self.relu1(self.norm1(self.conv1(x)))
    out = self.res_layer3(self.res_layer2(self.res_layer1(out)))
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


