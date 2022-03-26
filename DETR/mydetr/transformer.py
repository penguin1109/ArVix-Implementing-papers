import copy
import torch
from torch import tensor
import torch.nn as nn

from torch.nn import Embedding
# nn.Embedding allows to store word embeddings and retrieve them using indices
# 즉, Embedding(num_embeddings, embedding_dim)이라고 할때에 총 embedding된 sample의 개수가 num_embeddings이고
# 입력 값으로 우리는 원하는 샘플의 index를 전달하면 embedding dim의 길이로 embedding된 샘플의 해당하는 인덱스에 대해 반환

from torch.init import xavier_uniform_
# nn.MultiHeadAttention이라는 모듈을 사용해서 서로 다른 객체에 집중하는 attention network을 적용 가능하게 함
# Allows the model to jointly attend to information from different representation subspaces

from .layers import *

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm = None):
        super(TransformerEncoder, self).__init__()
        """
        self attention + feed forward + layer norm의 반복으로 이루어진 encoder
        """
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        return output
    
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 layer_norm_eps = 1e-5, batch_first = True, norm_first = False):
        # cpu인지 cuda환경인지 + float를 기본적으로 사용
        factory_kwargs = {'device':device, 'dtype':dtype}
        super(Transformer, self).__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout = 0.1, activation = 'gelu', 
                                               layer_norm_eps = layer_norm_eps, batch_first = batch_first, norm_first = norm_first,
                                               **factory_kwargs)
        encoder_norm = nn.LayerNorm(normalized_shape=d_model, eps = layer_norm_eps, **factory_kwargs)
        
        decoder_layer = TransformerDecoderLayer(d_model, eps = layer_norm_eps, **factory_kwargs)
        decoder_norm = nn.LayerNorm(normalized_shape = d_model, eps = layer_norm_eps, **factory_kwargs)
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        