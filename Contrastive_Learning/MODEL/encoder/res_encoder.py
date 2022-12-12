import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .blocks import PostResBlock

""" ResEncoder
- MOCO를 contrastive learning에 사용할 때에만 필요
"""
class ResEncoder(nn.Module):
  def __init__(self, hidden_dim=128):
    super(ResEncoder, self).__init__()
    ## (1) First Block used to generate the z, used for the sft layer when denoising 
    # The output of the first layer is used to preserve the features and the noise information at the same time
    self.first_block = PostResBlock(channel_in=1, channel_out=64, stride=1, conv_first=False) ## 첫번째 layer은 feature size를 입력과 동일햐게 사용
    ## (2) Second Block used to encode the Degradation Level
    self.encoder = nn.Sequential(
        PostResBlock(channel_in=64, channel_out=hidden_dim, stride=2, conv_first=False),
        PostResBlock(channel_in=hidden_dim, channel_out=hidden_dim*2, stride=2, conv_first=False),
        nn.ReLU(inplace = True), # nn.ELU() ##
    )
    ## (3) Final MLP Layer for the Latent Degradation Represenation Generation (일종의 projection layer임)
    """ ERROR FIX
    - 처음에는 계속해서 mlp의 ouput이 다른 input을 넣어줌에도 불구하고 동일한 값이 나왔었다.
    - 그래서 확인을 해 본 결과 encoder의 ouput에 Adaptive Avg Pool2d를 적용하니 값이 전부 일정한 벡터값이 나오는 것을 알 수 있었다.
    - 이는 encoder의 Output 결과가 반복된 Tanh activation function의 사용으로 인해서 -1 ~ 1사이의 값을 갖기 때문이다.
    """
    """
    self.mlp = nn.Sequential(
        nn.Linear(256,256),
        nn.LeakyReLU(0.1, True),
        nn.Linear(256,256),
    )
    """
    self.mlp = nn.Linear(hidden_dim*2, hidden_dim*2)
    self.hidden_dim = hidden_dim
    self._init_weight()

  def _init_weight(self):
    for name, m in self.named_modules():
      if isinstance(m, nn.Linear):
        bound = 1 / math.sqrt(m.weight.size(1))
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
          nn.init.uniform_(m.bias, -bound, bound)


  def forward(self, x):
    z = self.first_block(x)
    feat = self.encoder(z)
    feat_pool = F.adaptive_avg_pool2d(feat, output_size=1).squeeze(-1).squeeze(-1) ### (B, 256)
    out = self.mlp(feat_pool)

    return feat, out, z

