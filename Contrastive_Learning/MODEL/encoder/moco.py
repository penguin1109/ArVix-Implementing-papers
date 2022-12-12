import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from loguru import logger

class MoCo(nn.Module):
  def __init__(self, encoder, dim,
               training=True,
               K=256,
               m=0.999,
               T=0.07):
    """ MoCo
    a MoCo model with the query encoder, key encoder, and queue
    dim: Feature Dimension
    K: queue size
    m: moco momentum of updating the encoder
    T: softmax temperature
    """
    super(MoCo, self).__init__()
    self.training = training
    self.K = K
    self.m = m
    self.T = T
    self.query_encoder = encoder()
    self.key_encoder = encoder()

    for key_param, query_param in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
      key_param.data.copy_(query_param.data)
      key_param.requires_grad = False
    
    self.register_buffer("queue", torch.randn(dim, K))
    self.queue = F.normalize(self.queue, dim=0)
    self.register_buffer("queue_pointer", torch.ones(1, dtype = torch.long))
  
  @torch.no_grad()
  def _momentum_update(self):
    """
    update the parmeter weight of the encoder and key
    """
    for key_param, query_param in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
      key_param.data = key_param.data * self.m + (1. - self.m) * query_param.data

  @torch.no_grad()
  def _enqueue_dequeue(self, key_out):
    batch_size = key_out.shape[0]
    pointer = int(self.queue_pointer)
    self.queue[:, pointer:pointer+batch_size] = key_out.transpose(0,1)
    pointer = (pointer + batch_size) % self.K

    self.queue_pointer[0] = pointer
    

  def forward(self, query_img, key_img, training):
    self.training = training
    """ Inputs
    query_img: (B, C, H, W)
    key_img: (B, C, H, W)
    """
    if self.training:
      feature, query_out, inter = self.query_encoder(query_img)
      query_out = F.normalize(query_out, dim=1)

      with torch.no_grad():
        ## key계산할 때는 gradient update를 하지 못해야 함 (이번 Input을 parameter update에 사용하면 안되기 때문이다.)
        self._momentum_update()
        _, key_out, _ = self.key_encoder(key_img)
        key_out = F.normalize(key_out, dim=1)
      
      pos_logit = torch.einsum('nc,nc->n', [query_out, key_out]).unsqueeze(-1)
      neg_logit = torch.einsum('nc,ck->nk', [query_out, self.queue.clone().detach()])
      logit = torch.cat([pos_logit, neg_logit], dim=1)
      logit /= self.T

      labels = torch.zeros(logit.shape[0], dtype = torch.long).cuda() ## 전부 0이기 때문에 첫번째 embedding vector이 positive sample임을 의미하게 된다.
      self._enqueue_dequeue(key_out)
      logger.info(logit.shape)
      

      return feature, logit, labels, inter
    
    else:
      feature, query_out, inter = self.query_encoder(query_img)
      return feature, inter



    
