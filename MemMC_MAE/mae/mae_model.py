import torch
import torch.nn as nn
import numpy as np
from .layers import SelfAttention, FeedForward
from .encoder import EncoderBlock

class patch_embed(nn.Module):
  def __init__(self, img_size, patch_size, ch_in, embed_dim):
    super(patch_embed, self).__init__()
    self.conv = nn.Conv2d(ch_in, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.img_size = img_size
    self.patch_size = patch_size
    patch_n = int(self.img_size // self.patch_size)
    self.num_patches = int(patch_n * patch_n)

  def forward(self, x):
    B, C, H, W = x.shape
    
    x = self.conv(x) # [B, CH, patch_n, patch_n]
    x = x.flatten(2).transpose(1, 2) # [B, #patches, #embedding_dim]

    return x

class Block(nn.Module):
  ## Block used for the encoder of the vision transformer.
  ## For the ViTMAE, this block will be used both for the encoder and also the decoder.
  def __init__(self, dim, num_heads,
               mlp_ratio=4., qkv_bias=False, qk_norm=True, proj_drop=0., attn_drop=0., use_proj_conv=False,
               act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super(Block, self).__init__()
    self.norm1 = norm_layer(dim)
    self.self_attention = SelfAttention(
        embedding_dim=dim, num_heads=num_heads, qk_norm=qk_norm, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
    ) # Multi Head Attention

    self.norm2 = norm_layer(dim)
    self.mlp = FeedForward(in_dim=dim, mid_dim=int(dim * mlp_ratio), out_dim=dim, drop_rate=proj_drop, use_conv=use_proj_conv) # Multi Layer Perceptron

    self.drop1 = nn.Dropout(attn_drop)
    self.drop2 = nn.Dropout(proj_drop)
  
  def forward(self, x):
    # residual connection
    x = x + self.drop1(self.self_attention(self.norm1(x)))
    
    x = x + self.drop2(self.mlp(self.norm2(x)))

    return x


class AnnoMAE(nn.Module):
  def __init__(self, img_size=224, patch_size=16, ch_in=3, 
               encoder_embed_dim=1024, encoder_depth=8, encoder_num_heads=8,
               decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
               mlp_ratio=4., norm_layer=nn.LayerNorm):
    super(AnnoMAE, self).__init__()
    # -------------------------------------------------------------------------------------------------------------------
    # AnnoMAE Encoder #
    self.patch_embed = patch_embed(img_size, patch_size, ch_in, encoder_embed_dim)
    self.patch_size = patch_size
    num_patches = self.patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros((1, 1, encoder_embed_dim)))
    # 앞에 class token이 있어야 하기 때문에 1 + num_patches로 사용 - fixed position embedding vector
    self.encoder_pos_embed = nn.Parameter(torch.zeros((1, 1 + num_patches, encoder_embed_dim)), requires_grad=False)

    self.encoder = nn.ModuleList([
        EncoderBlock(
            ch_in=ch_in, dim=encoder_embed_dim, num_heads=encoder_num_heads
        ) for _ in range(encoder_depth)
    ])
    self.encoder_norm = norm_layer(encoder_embed_dim)

    # ---------------------------------------------------------------------------------------------------------------------
    # AnnoMAE Decoder #
    self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
    self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True) # decoder에 입력으로 넣어주고자 하는 encoder의 output의 embedding dimension을 맞춰주고자 한다.

    self.decoder = nn.ModuleList([
        Block(
            dim=decoder_embed_dim, num_heads=decoder_num_heads
        # DecoderBlock(
          #  dim=decoder_embed_dim, # k_dim=encoder_embed_dim, 
          #  k_dim=decoder_embed_dim,
          #  num_heads=decoder_num_heads
        ) for _ in range(decoder_depth)
    ])

    self.decoder_pos_embed = nn.Parameter(torch.zeros((1, 1 + num_patches, decoder_embed_dim)), requires_grad=False)
    self.decoder_norm = norm_layer(decoder_embed_dim)
    self.decoder_proj = nn.Linear(decoder_embed_dim, patch_size**2*ch_in, bias=True)
    
    # Weight Inintialization
    self.initialize_weights()
    self.apply(self._init_weights)

  def sinusoidal_pe(self, pos_embed):
    B, L, E = pos_embed.shape
    for pos in range(L):
      for i in range(0, E-1, 2):
        pos_embed[:, pos,i] = np.sin(pos / 10000 ** (i / E))
        pos_embed[:, pos, i+1] = np.cos(pos / 10000 ** (i / E))
        # pos_embed[:, pos, i] = torch.sin(torch.Tensor([10000 ** (i / E)]))
        # pos_embed[:, pos, i+1] = torch.cos(torch.Tensor([10000 ** (i / E)]))
    return pos_embed

  def initialize_weights(self):
    # class token을 position으로 간주를 하면 안됨! 
    encoder_pos_embed = np.zeros(tuple(self.encoder_pos_embed.shape))
    encoder_pos_embed = np.concatenate((np.expand_dims(encoder_pos_embed[:, 0, :], 1), self.sinusoidal_pe(encoder_pos_embed[:, 1:, :])), 1)
    self.encoder_pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float())
    decoder_pos_embed = np.zeros(tuple(self.decoder_pos_embed.shape))
    decoder_pos_embed = np.concatenate((np.expand_dims(decoder_pos_embed[:, 0, :], 1), self.sinusoidal_pe(decoder_pos_embed[:, 1:, :])), 1)
    self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float())
    
    # self.encoder_pos_embed = torch.cat((torch.unsqueeze(self.encoder_pos_embed[:, 0, :], 1), self.sinusoidal_pe(self.encoder_pos_embed[:, 1:, :])), 1)
    # self.decoder_pos_embed = torch.cat((torch.unsqueeze(self.decoder_pos_embed[:, 0, :], 1), self.sinusoidal_pe(self.decoder_pos_embed[:, 1:, :])), 1)

    w = self.patch_embed.conv.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    nn.init.normal_(self.cls_token, std=.02)
    nn.init.normal_(self.mask_token, std=.02)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1)

  def patchify(self, imgs):
    # imgs: [B, ch_in, H, W]
    B, C, H, W = imgs.shape
    p = self.patch_size
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape = (imgs.shape[0], C, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h*w, p*p*C))

    return x
    
  def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    # x is the generated token for each input patch using linear projection and an added positional embedding
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))        
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
    
  def forward_encoder(self, x, mask_ratio=0.75):
    x = self.patch_embed(x)

    # summation of the position embedding vector
    x = x + self.encoder_pos_embed[:, 1:, :]

    x, mask, ids_restore = self.random_masking(x, mask_ratio)

    cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    for encoder in self.encoder:
      x = encoder(x)
    x = self.encoder_norm(x)

    # The encoder output consists of a CLS token and a series of feature tokens.
    # CLS token: captures global image information
    # Feature token: represents local and global information for different patches
    return x, mask, ids_restore
  
  def forward_decoder(self, x, ids_restore):
    """
    1. Input of the decoder is the full set of tokens of (1) encoded visible patches + (2) Mask Tokens
    2. Each mask token is a shared, learned vector 
    """
    encoder_input = x
    x = self.decoder_embed(x)
    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) # masking되어 있지 않던 token들과 합해줌 (단, 이때 dummy token임)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
    x = torch.cat([x[:, :1, :], x_], dim=1) # concatenate cls token 

    x = x + self.decoder_pos_embed

    for decoder in self.decoder:
      # x = decoder(x, encoder_input, encoder_input)
      # x = decoder(x, x, x)
      x = decoder(x)

    
    x = self.decoder_norm(x)
    x = self.decoder_proj(x)

    # remove class token
    x = x[:, 1:, :]

    return x

  def forward_loss(self, imgs, pred, mask):
    """
    imgs: [B, 3, H, W]
    pred: [B, L, p*p*3]
    mask: [B, L], 0 is keep, 1 is remove, 
    """
    target = self.patchify(imgs) # [B, L, E]

    loss = (pred - target) ** 2 # MSE Loss 
    loss = loss.mean(dim=-1)  # [B, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

  def forward(self, x, mask_ratio=0.75):
    latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
    pred = self.forward_decoder(latent, ids_restore)
    loss = self.forward_loss(x, pred, mask)

    return loss, pred, mask