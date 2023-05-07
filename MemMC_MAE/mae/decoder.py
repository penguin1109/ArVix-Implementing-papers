import torch.nn as nn
from .layers import SelfAttention, FeedForward

class DecoderBlock(nn.Module):
  def __init__(self, dim, k_dim, num_heads,
               mlp_ratio=4., qkv_bias=False, qk_norm=True, proj_drop=0.,
               attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super(DecoderBlock, self).__init__()
    self.self_attention = SelfAttention(
        embedding_dim=dim, num_heads=num_heads, qk_norm=qk_norm, qkv_bias=qkv_bias
    )
    self.cross_attention = CrossAttention(
        embedding_dim=dim, k_dim=k_dim, num_heads=num_heads, qk_norm=qk_norm, qkv_bias=qkv_bias
    )
    self.feed_forward = FeedForward(in_dim=dim, mid_dim=int(dim*mlp_ratio), out_dim=dim, act_layer=act_layer, norm_layer=norm_layer)

    self.norm1 = norm_layer(dim)
    self.norm2 = norm_layer(dim)
    self.norm3 = norm_layer(dim)

    self.drop1 = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()
    self.drop2 = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()
    self.drop3 = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

  
  def forward(self, q, k, v):
    B, N, E = q.shape
    q = q + self.drop1(self.self_attention(self.norm1(q)))

    x = q + self.drop2(self.cross_attention(self.norm2(q), k, v))

    x = x + self.drop3(self.feed_forward(self.norm3(x)))

    return x