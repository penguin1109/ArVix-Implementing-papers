import torch
import torch.nn as nn

__all__ = ['HeartSarUNet'] # UNet, nested improved UNet인 deep supervision을 사용하는 UNet++, 그리고 ASPPResUnet의 구현이다.

class SEBlock(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SEBlock, self).__init__()
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    #print(y.expand_as(x).shape)

    return x * y.expand_as(x)

class ASPPmodule(nn.Module):
  def __init__(self, channel_in, channel_out, momentum, dilation):
    super(ASPPmodule, self).__init__()
    self.conv = nn.Conv2d(channel_in, channel_out, kernel_size = 3,padding = dilation, dilation = dilation)
    self.relu = nn.ReLU(inplace = True)
    self.bn = nn.BatchNorm2d(num_features = channel_out, momentum = momentum)
  
  def forward(self, x):
    out = self.conv(x)
    out = self.relu(out)
    out = self.bn(out)

    return out

class ASPP(nn.Module):
  def __init__(self, channel_in, channel_mid, channel_out, momentum = 0.0003, dilation= [6,12,18]):
    """
    channel_in은 입력 값의 channel 개수
    channel_out은 출력 값의 channel의 개수 (4개를 concat해서 1024를 만들어야 하는 상황이기 때문에 256로 설정)
    """
    super(ASPP, self).__init__()
    self.aspp_1 = ASPPmodule(channel_in, channel_mid, momentum, dilation[0])
    self.aspp_2 = ASPPmodule(channel_in, channel_mid, momentum, dilation[1])
    self.aspp_3 = ASPPmodule(channel_in, channel_mid, momentum, dilation[2])
    #self.aspp_4 = ASPPmodule(channel_in, channel_mid, momentum, dilation[3])

    self.conv = nn.Conv2d(channel_mid * len(dilation), channel_out, kernel_size = 1)
  
  def forward(self, x):
    out_1 = self.aspp_1(x)
    out_2 = self.aspp_2(x)
    out_3 = self.aspp_3(x)
    #out_4 = self.aspp_4(x)

    out = torch.cat((out_1, out_2, out_3), 1)
    out = self.conv(out)
    
    return out

class ResBlock(nn.Module):
  def __init__(self, channel_in, channel_out):
    super(ResBlock, self).__init__()
    self.conv_1 = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(inplace = True)
    )
    self.conv_2 = nn.Sequential(
        nn.Conv2d(channel_out, channel_out, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(channel_out)
    )
    self.relu = nn.ReLU(inplace = True)
    self.skip = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size = 1, padding = 0, stride = 1),
        nn.BatchNorm2d(channel_out)
    )
  
  def forward(self, x):
    #print(f"res block input shape : {x.shape}")
    out = self.conv_1(x) #(N, channel_in, H, W)
    #print(f"res block out1 shape : {out.shape}")
    out = self.conv_2(out) #(N, channel_out, H, W)


    x = self.skip(x) #(N, channel_in, H, W)
    #print(out.shape, x.shape)
    return self.relu(out + x) #(N, channel_out, H, W)

class CSEBlock(nn.Module):
  def __init__(self, channel_in, channel_mid, channel_out):
    super(CSEBlock, self).__init__()
    self.res = ResBlock(channel_in = channel_in, channel_out = channel_mid)
    self.se = SEBlock(channel = channel_mid)

  def forward(self, x):
    out = self.res(x)
    out = self.se(out)

    return out


class HeartSarUnet(nn.Module):
  def __init__(self, num_classes, first_channel, channel_in_start = 12, channel_mid_start = 4):
    super(HeartSarUnet, self).__init__()
    self.down_channels = [channel_in_start * (2**(i-1)) for i in range(1,5)]
    self.down_channels = [first_channel]+self.down_channels # [first_channel, 64, 128, 256, 512] -> [first_channel, 8, 16, 32, 64]
    #print(self.down_channels)
    self.mid_channels = [channel_mid_start * (2**i) for i in range(4)] # [4, 8, 16, 32]
    #print(self.mid_channels)
    self.up_channels = [channel_in_start * (2**i) for i in range(4, -1, -1)] # [1024, 512, 256, 128, 64] -> [128, 64, 32 16, 8]
    #print(self.up_channels)

    self.cat_channels = [self.down_channels[4-i]+self.up_channels[i] for i in range(5)] # [1536, 756, 384, 192]
    #print(self.cat_channels)
    self.down_1 = CSEBlock(channel_in = self.down_channels[0], channel_mid = self.down_channels[1], channel_out = self.mid_channels[0]) # (0, 64, 4)
    self.down_2 = CSEBlock(channel_in = self.down_channels[1], channel_mid = self.down_channels[2], channel_out = self.mid_channels[1]) # (64, 128, 8)
    self.down_3 = CSEBlock(channel_in = self.down_channels[2], channel_mid = self.down_channels[3], channel_out = self.mid_channels[2]) # (128, 256, 16)
    self.down_4 = CSEBlock(channel_in = self.down_channels[3], channel_mid = self.down_channels[4], channel_out = self.mid_channels[3]) # (256, 512, 32)
    self.max_pool = nn.MaxPool2d(kernel_size = 2)

    self.ASPP_1 = ASPP(channel_in = self.down_channels[4], channel_mid = self.down_channels[4], channel_out = self.up_channels[0])

    self.upsample = nn.Upsample(scale_factor = 2)
    self.up_1 = ResBlock(channel_in = self.cat_channels[0], channel_out = self.up_channels[1])
    self.up_2 = ResBlock(channel_in = self.cat_channels[1], channel_out = self.up_channels[2])
    self.up_3 = ResBlock(channel_in = self.cat_channels[2], channel_out = self.up_channels[3])
    self.up_4 = ResBlock(channel_in = self.cat_channels[3], channel_out = self.up_channels[4])

    self.ASPP_2 = ASPP(channel_in = self.up_channels[4], channel_mid = self.up_channels[4], channel_out = self.up_channels[4])

    self.final_conv = nn.Conv2d(self.up_channels[4], num_classes, kernel_size = 1)



  def forward(self, x):
    ###### Encoder ######
    se_1 = self.down_1(x) # concat with up_4
    down_1 = self.max_pool(se_1)
    #print(down_1.shape)

    se_2 = self.down_2(down_1) # concat with up_3
    down_2 = self.max_pool(se_2)

    se_3 = self.down_3(down_2) # concat with up_2
    down_3 = self.max_pool(se_3)

    se_4 = self.down_4(down_3) # concat with up_1
    down_4 = self.max_pool(se_4)

    ###### Bridge #####
    aspp_1 = self.ASPP_1(down_4)
    #print(f"aspp_1 shape : {aspp_1.shape}")

    ##### Decoder #######
    up4_1 = self.upsample(aspp_1)
    cat_4 = torch.cat((up4_1, se_4), dim = 1)
    up4_2 = self.up_1(cat_4)

    up3_1 = self.upsample(up4_2)
    cat_3 = torch.cat((up3_1, se_3), dim = 1)
    up3_2 = self.up_2(cat_3)

    up2_1 = self.upsample(up3_2)
    cat_2 = torch.cat((up2_1, se_2), dim = 1)
    up2_2 = self.up_3(cat_2)

    up1_1 = self.upsample(up2_2)
    cat_1 = torch.cat((up1_1, se_1), dim = 1)
    up1_2 = self.up_4(cat_1)

    ##### FInal ####
    aspp_2 = self.ASPP_2(up1_2)
    result = self.final_conv(aspp_2)

    return result



    