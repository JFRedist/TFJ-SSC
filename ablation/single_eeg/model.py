import math

import torch
from torch import nn
from torch.autograd import Variable
from relative_position import RelativePositionalEncoding, RelativeMultiHeadAttention
from relative_transformer import RelativeTransformerEncoderLayer, RelativeTransformerEncoder


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model=128, dropout=0.2, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe:[1, 30, 128]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        # Choose between absolute and relative positional encoding
        if hasattr(config, 'use_relative_pos') and config.use_relative_pos:
            self.position_single = RelativePositionalEncoding(
                d_model=config.dim_model, 
                dropout=0.1,
                max_relative_position=config.max_relative_position
            )
            # Custom encoder layer with relative positional encoding
            self.use_relative_pos = True
            
            # Use custom transformer encoder with relative positional encoding
            rel_encoder_layer = RelativeTransformerEncoderLayer(
                d_model=config.dim_model, 
                nhead=config.num_head, 
                dim_feedforward=config.forward_hidden, 
                dropout=config.dropout,
                max_relative_position=config.max_relative_position,
                batch_first=True
            )
            self.transformer_encoder_1 = RelativeTransformerEncoder(rel_encoder_layer, num_layers=config.num_encoder)
            self.transformer_encoder_2 = RelativeTransformerEncoder(rel_encoder_layer, num_layers=config.num_encoder)
            self.transformer_encoder_3 = RelativeTransformerEncoder(rel_encoder_layer, num_layers=config.num_encoder)
        else:
            self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)
            self.use_relative_pos = False
            
            # Standard transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head, dim_feedforward=config.forward_hidden, dropout=config.dropout, batch_first=True)
            self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
            self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
            self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        # Choose between absolute and relative positional encoding for multi-channel
        if hasattr(config, 'use_relative_pos') and config.use_relative_pos:
            self.position_multi = RelativePositionalEncoding(
                d_model=config.dim_model * 3, 
                dropout=0.1,
                max_relative_position=config.max_relative_position
            )
            
            # Use custom transformer encoder with relative positional encoding for multi-channel
            rel_encoder_layer_multi = RelativeTransformerEncoderLayer(
                d_model=config.dim_model * 3, 
                nhead=config.num_head, 
                dim_feedforward=config.forward_hidden, 
                dropout=config.dropout,
                max_relative_position=config.max_relative_position,
                batch_first=True
            )
            self.transformer_encoder_multi = RelativeTransformerEncoder(rel_encoder_layer_multi, num_layers=config.num_encoder_multi)
        else:
            self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)
            
            # Standard transformer encoder layer for multi-channel
            encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model * 3, nhead=config.num_head,dim_feedforward=config.forward_hidden, dropout=config.dropout, batch_first=True)
            self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)

        self.fc1 = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model * 3, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

    def forward(self, x):
        # 只使用单个EEG通道(Fpz-Cz)
        x1 = x[:, 0, :, :]
        # Apply positional encoding
        if self.use_relative_pos:
            # For relative positional encoding, we get both the input tensor and position embeddings
            x1, rel_embeddings1 = self.position_single(x1)
            # The relative position embeddings will be used in custom attention mechanisms
            # but for now we'll continue with the standard transformer encoder
        else:
            # For absolute positional encoding, we directly add the embeddings
            x1 = self.position_single(x1)

        x1 = self.transformer_encoder_1(x1)     # (batch_size, 29, 128)
        
        # 复制x1三次以保持与原始模型相同的维度
        x = torch.cat([x1, x1, x1], dim=2)

        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        # Apply multi-channel positional encoding
        if self.use_relative_pos:
            # For relative positional encoding in multi-channel
            x, _ = self.position_multi(x)
            # Use custom transformer encoder with relative positional encoding
            x = self.transformer_encoder_multi(x)
        else:
            # For absolute positional encoding
            x = self.position_multi(x)
            x = self.transformer_encoder_multi(x)

        x = self.layer_norm(x + residual)       # residual connection

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x