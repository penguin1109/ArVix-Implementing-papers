import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == 'gelu':
        return F.gelu
    elif activation == 'relu':
        return F.relu
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout = 0.1, activation = 'gelu', layer_norm_eps = 1e-5,
                 batch_first = True, norm_first = False, device = None, dtype = None):
        super(TransformerEncoderLayer, self).__init__()
        factory_kwargs = {'device' : device, 'dtype' : dtype}
        """
        d_model : 입력 feature의 개수
        n_heads : multiheadattention에서 사용될 head의 개수
        dim_feedforward : feedforward network model에서 사용될 dimension의 크기
        """
        self.self_attn = nn.MultiHeadAttention(d_model, dim_feedforward, **factory_kwargs)
        # Feed Forward Network
        self.lin1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(p = dropout)
        self.lin2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(dim_feedforward, eps = layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(TransformerEncoderLayer, self).__setstate__(state)

        def forward(self, src, src_mask = None, src_key_padding_mask = None):
            """
            src : encoder layer에 입력으로 넣어줄 sequential data
            """
            x = src
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(self._ff_block(x))
            return x
    
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask = attn_mask, key_padding_mask = key_padding_mask)[0]
        return self.dropout1(x)
    def _ff_block(self, x):
        x = self.lin2(self.dropout(self.activation(self.lin1(x))))
        return x
            
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout, activation = F.gelu, 
                 layer_norm_eps = 1e-5, batch_first = False, norm_first = False, device = None, dtype = None):
        factory_kwargs = {'device' : device, 'dtype' : dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, n_head, dropout = dropout, batch_first = batch_first, **factory_kwargs)
        self.multihead_attn = nn.MultiHeadAttention(d_model, n_head, dropout = dropout, batch_first = batch_first, **factory_kwargs)
        # feed forward network
        self.lin1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps)