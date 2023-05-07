import torch.nn as nn

class FeedForward(nn.Module):
  def __init__(self, in_dim, mid_dim, out_dim, drop_rate=0.,
               act_layer=nn.GELU, norm_layer=None, 
               bias=True, use_conv=False):
    super(FeedForward, self).__init__()
    # Dim -> Dim * ratio -> Dim
    if use_conv:
      self.fc1 = nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=bias)
      self.fc2 = nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=bias)
    else:
      self.fc1 = nn.Linear(in_dim, mid_dim, bias=bias)
      self.fc2 = nn.Linear(mid_dim, out_dim, bias=bias)
    
    self.act = act_layer()
    self.norm = norm_layer(mid_dim) if norm_layer else nn.Identity()
    self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()
    self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()
  
  def forward(self, x):
    x = self.drop1(self.act(self.fc1(x)))
    x = self.drop2(self.fc2(x))

    return x

def scaled_dot_product_attention(q, k, v, scale, attn_drop):
  q = q * scale
  attn = q @ k.transpose(-2, -1) # Q(K^T)
  attn = attn.softmax(dim=-1)
  attn = attn_drop(attn)

  x = attn @ v
  
  return x

class SelfAttention(nn.Module):
  """
  Multi Head Attention used in the encoder of the transformer model.
  Uses the encoder input for all query, key, and value used in the dot-product attention.
  """
  def __init__(self, embedding_dim, num_heads,
               qk_norm, qkv_bias, attn_drop=0., proj_drop=0.,
               norm_layer = nn.LayerNorm):
    super(SelfAttention, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    assert embedding_dim % num_heads == 0
    self.dim_head = embedding_dim // num_heads # MHA에서 하나의 head의 dimension
    self.scale = self.dim_head ** (-0.5)

    ## QKV Constraints ##
    self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
    self.q_norm = norm_layer(self.dim_head) if qk_norm else nn.Identity()
    self.k_norm = norm_layer(self.dim_head) if qk_norm else nn.Identity()

    self.proj = nn.Linear(embedding_dim, embedding_dim)
    self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
    self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()
  
  def forward(self, x):
    B, L, E = x.shape
    qkv = self.qkv(x)
    qkv = qkv.reshape(B, L, 3, self.num_heads, self.dim_head)
    qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, #head, L, dim_head]
    q, k, v = qkv.unbind(0) # [B, #head, L, dim_head]
    q, k = self.q_norm(q), self.k_norm(k)

    x = scaled_dot_product_attention(q, k, v, self.scale, self.attn_drop) # [B, #head, L, dim_head]
    x = x.transpose(1, 2).reshape(B, L, E)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x

class CrossAttention(nn.Module):
  """
  Multi Head Attention used for the decoder of the transformer model.
  Uses the encoder output as the key, value and the decoder dimension embedded input vector as the query.
  """
  def __init__(self, embedding_dim, k_dim, num_heads, 
               qk_norm, qkv_bias, attn_drop=0., proj_drop=0.,
               norm_layer=nn.LayerNorm):
    super(CrossAttention, self).__init__()
    self.q_dim = embedding_dim
    self.k_dim = k_dim
    self.v_dim = k_dim

    self.q_fc = nn.Linear(self.q_dim, self.q_dim)
    self.k_fc = nn.Linear(self.k_dim, self.q_dim)
    self.v_fc = nn.Linear(self.v_dim, self.q_dim)

    self.proj = nn.Linear(self.q_dim, self.q_dim)
  
    self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
    self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

    self.num_heads = num_heads
    self.dim_heads = embedding_dim // num_heads
    self.scale = self.dim_heads ** (-0.5) # scale을 곱해줘야 하니까 -0.5의 제곱을 계산하는것이 맞다.

    self.q_norm = norm_layer(self.dim_heads) if qk_norm else nn.Identity()
    self.k_norm = norm_layer(self.dim_heads) if qk_norm else nn.Identity()
  
  def forward(self, q, k, v):
    B, QN, E = q.shape
    KN, VN = k.shape[1], v.shape[1]

    q = self.q_fc(q) # [B, QN, E]
    q = q.reshape(B, QN, self.num_heads, self.dim_heads) # E = self.num_heads * self.head_dim 
    q = q.permute(0, 2, 1, 3)
    q = self.q_norm(q)
    
    k = self.k_norm(self.k_fc(k).reshape(B, KN, self.num_heads, self.dim_heads).permute(0, 2, 1, 3))
    v = self.v_fc(v).reshape(B, VN, self.num_heads, self.dim_heads).permute(0, 2, 1, 3)

    x = scaled_dot_product_attention(q, k, v, self.scale, self.attn_drop) # [B, #head, N, #dim]
  
    x = x.transpose(1, 2) # [B, N, #head, #dim]
    x = x.reshape(B, QN, E) # [B, N, E]

    x = self.proj(x)
    x = self.proj_drop(x)

    return x
                                                                  