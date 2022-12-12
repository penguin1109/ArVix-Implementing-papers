import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import PostResBlock

class SimSiam(nn.Module):
  def __init__(self, hidden_dim, pred_dim):
    super(SimSiam, self).__init__()
    self.first_block = PostResBlock(channel_in=1, channel_out=64, stride=1, conv_first=False)
    self.hidden_dim = hidden_dim
    self.pred_dim = pred_dim
    # FINAL ENCODER LAYER
    self.encoder = nn.Sequential(
        PostResBlock(channel_in=64, channel_out=hidden_dim, stride=2, conv_first=False),
        PostResBlock(channel_in=hidden_dim, channel_out=hidden_dim*2, stride=2, conv_first=False),
        nn.ReLU(inplace = True), # nn.ELU() ##
      )
    ## END OF THE ENCODER (BASE ENCODER)
    # A PROJECTION MLP LAYER -> 원래대로라면 batch norm의 크기를 매우 크게 잡아야 한다.
    prev_dim = hidden_dim*2
    self.projection = nn.Sequential(
        nn.Linear(prev_dim, prev_dim, bias=False),
        nn.BatchNorm1d(prev_dim),
        nn.ReLU(inplace = True),
        nn.Linear(prev_dim, prev_dim, bias=False),
        nn.BatchNorm1d(prev_dim),
        nn.ReLU(inplace = True),
        nn.Linear(prev_dim, pred_dim, bias= False),
        nn.BatchNorm1d(pred_dim, affine=False)
    )

    # A PREDICTION MLP LAYER
    self.mlp = nn.Sequential(
        nn.Linear(pred_dim, pred_dim * 4, bias=False),
        nn.BatchNorm1d(pred_dim*4),
        nn.ReLU(inplace = True),
        nn.Linear(pred_dim*4, pred_dim)
    )

  def forward(self, query_x, key_x, training=True):
    if training == False:
      z = self.first_block(query_x)
      feature = self.encoder(z)
      feature = F.adaptive_avg_pool2d(feature, output_size=1).squeeze(-1).squeeze(-1)
      proj = self.projection(feature)
      pred = self.mlp(proj)
      return z, proj, pred

    else: ## qury와 key모두에 대한 output을 만들어야 함
      z = self.first_block(query_x)
      feature_query, feature_key = self.encoder(z), self.encoder(self.first_block(key_x))
      feature_query = F.adaptive_avg_pool2d(feature_query, output_size=1).squeeze(-1).squeeze(-1)
      feature_key = F.adaptive_avg_pool2d(feature_key, output_size=1).squeeze(-1).squeeze(-1)

      proj_query, proj_key = self.projection(feature_query), self.projection(feature_key)
      pred_query, pred_key = self.mlp(proj_query), self.mlp(proj_key)

      ## detach to stop the gradient flow
      return z, proj_query.detach(), proj_key.detach(), pred_query, pred_key
