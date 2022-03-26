## bottleneck ResNet v2 with GroupNorm and Weight Standardization

def conv3x3(cin, cout, stride= 1, groups = 1, bias = False):
  return StdConv2d(cin, cout, kernel_size = 3, stride = stride, padding = 1, bias = bias, groups = groups)

def conv1x1(cin, cout, stride = 1, groups = 1, bias = False):
  return StdConv2d(cin, cout, kernel_size = 1, stride = stride, padding = 0, bias = bias, groups = groups)

class PreActBottleneck(nn.Module):
  def __init__(self, cin, cout = None, cmid = None, stride = 1):
    super().__init__()
    cout = cout or cin
    cmid = cmid or cout // 4

    self.gn1 = nn.GroupNorm(32, cin)
    self.conv1 = conv1x1(cin, cmid)
    self.gn2 = nn.GroupNorm(32, cmid, stride)
    self.conv2 = conv3x3(cmid, cmid)
    self.gn3 = nn.GroupNorm(32, cmid)
    self.conv3 = conv1x1(cmid, cout)
    self.relu = nn.ReLU(inplace = True) ## 일반적으로 relu는 하나만 구현을 하는 편이다.

    # projection also with pre-activation
    if (stride != 1 or cin != cout):
      self.downsample = conv1x1(cin, cout, stride)

  def forward(self, x):
    out = self.relu(self.gn1(x))

    residual = x
    if hasattr(self, 'downsample'): 
      residual = self.downsample(out)
    out = self.conv1(out)
    out = self.conv2(self.relu(self.gn2(out)))
    out = self.conv3(self.relu(self.gn3(out)))

    return out + residual
