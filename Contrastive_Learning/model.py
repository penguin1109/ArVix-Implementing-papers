import torch
import torch.nn as nn
import torch.nn.functional as F
from MODEL.denoiser.dgrn import DGRN
from MODEL.encoder.cbde import CBDE

""" Gauss-AIRNet
- DGM 부분에서 GaussNet의 Post Res Block을 사용하게 된다.
- CBDE 부분에서는 MoCo를 기반으로 contrastive learning을 위한 query image와 key image의 embedding 값을 구하게 된다.
"""
class GaussAIRNet(nn.Module):
  def __init__(self, batch_size, moco=False, training=True):
    super(GaussAIRNet, self).__init__()
    """ GaussAirNet
    - moco: bool (tells you if the encoder uses the MoCo of SimSiamese as the contrastive learning approach)
      - the objective of the encoder is to learn the NOISE REPRESENTATIVE of the image
    """
    self.moco = moco
    self.RESTORER = DGRN(channel_in=1, channel_out=1, feat=64, group_n=5)
    self.ENCODER = CBDE(batch_size=batch_size, moco=self.moco)

  def forward(self, query_x, key_x, training):
    self.training = training
    if self.training: ## key_x는 positive sample이어야 한다. (예를 들면 query_x에서 증강하여 얻은 데이터)
      if self.moco == False:
        inter, query_feat, key_feat, query_cls, key_cls = self.ENCODER(query_x, key_x, training = True)
        restored = self.RESTORER(query_x, inter)
        return restored, query_feat, key_feat, query_cls, key_cls
      else:
        feature, logit, label, inter = self.ENCODER(query_x, key_x, training=training) ## denoiser로 사용하려는 기본적인 모델
        restored = self.RESTORER(query_x, inter)
        return restored, logit, label
    else:
      if self.moco == False:
        inter, query_cls = self.ENCODER(query_x, key_x, training = False)
        restored = self.RESTORER(query_x, inter)
        return restored
      else:
        feature, inter = self.ENCODER(query_x, query_x, training = training) ## prediction 단계에서는 어차피 param update를 안하기 때문에 query = key이미지로 사용
        restored = self.RESTORER(query_x, inter)
        return restored
