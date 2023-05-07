import torch.nn as nn
from .layers import SelfAttention, FeedForward

class EncoderBlock(nn.Module):
  def __init__(self, ch_in, 
               dim, num_heads, mlp_ratio=4.,
               qkv_bias=False, qk_norm=True, proj_drop=0., attn_drop=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super(EncoderBlock, self).__init__()
    self.ch_in = ch_in
    self.dim = dim
    self.num_heads = num_heads

    self.norm1 = norm_layer(dim)
    self.drop1 = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()
    self.drop2 = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

    self.mha = SelfAttention(
        embedding_dim=dim, num_heads=num_heads, qk_norm=qk_norm, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer
    )
    self.norm2 = norm_layer(dim)

    self.feed_forward = FeedForward(in_dim=dim, mid_dim=int(dim * mlp_ratio), out_dim=dim, act_layer=act_layer, norm_layer=norm_layer)

  
  def forward(self, x):
    x = x + self.drop1(self.mha(self.norm1(x)))
    x = x + self.drop2(self.feed_forward(self.norm2(x)))

    return x

