import torch
import torch.nn as nn

__all__ = ['UNet', 'UNetPP', 'ARUNet'] # UNet, nested improved UNet인 deep supervision을 사용하는 UNet++, 그리고 ASPPResUnet의 구현이다.

class VGGBlock(nn.Module):
  def __init__(self, channel_in, channel_mid, channel_out):
    super().__init__()
    self.relu = nn.ReLU(inplace = True)
    self.conv1 = nn.Conv2d(in_channels = channel_in, out_channels = channel_mid, kernel_size = 3, padding = 1)
    self.bn1 = nn.BatchNorm2d(channel_mid)
    self.conv2 = nn.Conv2d(in_channels = channel_mid, out_channels = channel_out, kernel_size = 3, padding = 1)
    self.bn2 = nn.BatchNorm2d(channel_out)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    out = self.relu(x)

    return out

class UNet(nn.Module):
  def __init__(self, num_classes, channel_in = 3, start_filter = 16):
    super().__init__()

    self.filters = []
    for i in range(5):
      self.filters.append(start_filter*(2**i))
    
    self.pool = nn.MaxPool2d(2,2)
    self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)

    self.conv0_0 = VGGBlock(channel_in, self.filters[0], self.filters[0])
    self.conv1_1 = VGGBlock(self.filters[0], self.filters[1], self.filters[1])
    self.conv2_1 = VGGBlock(self.filters[1], self.filters[2], self.filters[2])
    self.conv3_1 = VGGBlock(self.filters[2], self.filters[3], self.filters[3])
    self.conv4_1 = VGGBlock(self.filters[3], self.filters[4], self.filters[4])

    self.conv4_2 = VGGBlock(self.filters[4] + self.filters[3], self.filters[3], self.filters[3])
    self.conv3_2 = VGGBlock(self.filters[3]+self.filters[2], self.filters[2], self.filters[2])
    self.conv2_2 = VGGBlock(self.filters[2] + self.filters[1], self.filters[1], self.filters[1])
    self.conv1_2 = VGGBlock(self.filters[1] + self.filters[0], self.filters[0], self.filters[0])

    # kernel_size = 1,즉 1x1의 convolution layer을 사용해서 Linear layer을 사용한 것과 동일한 효과를 가져온다
    self.out = nn.Conv2d(self.filters[0], num_classes, kernel_size = 1) 

  def forward(self, x):
    x0_0 = self.conv0_0(x)
    x1_0 = self.conv1_1(self.pool(x0_0))
    x2_0 = self.conv2_1(self.pool(x1_0))
    x3_0 = self.conv3_1(self.pool(x2_0))
    x4_0 = self.conv4_1(self.pool(x3_0))

    x3_1 = self.conv4_2(torch.cat([x3_0, self.up(x4_0)], 1))
    x2_2 = self.conv3_2(torch.cat([x2_0, self.up(x3_1)], 1))
    x1_3 = self.conv2_2(torch.cat([x1_0, self.up(x2_2)], 1))
    x0_4 = self.conv1_2(torch.cat([x0_0, self.up(x1_3)], 1))

    out = self.out(x0_4)
    return out

class UNetPP(nn.Module):
    def __init__(self, num_classes, deep_supervision = False, channel_in = 3, start_filter = 8):
        super().__init__()

        nb_filter = []
        for i in range(5):
            nb_filter.append(start_filter * (2**i))

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(channel_in, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


    