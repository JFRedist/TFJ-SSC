import torch
import torch.nn as nn
import torch.nn.functional as F
from relative_position import RelativeMultiHeadAttention

class RelativeTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Relative Positional Encoding.
    
    This implementation is based on the paper 'Self-Attention with Relative Position Representations'
    by Shaw et al. (2018) and adapts the standard PyTorch TransformerEncoderLayer to use
    relative positional encoding.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 max_relative_position=20, batch_first=True):
        super(RelativeTransformerEncoderLayer, self).__init__()
        
        # Multi-head attention with relative positional encoding
        self.self_attn = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            max_relative_position=max_relative_position
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
        self.batch_first = batch_first
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Forward pass for the relative transformer encoder layer.
        
        Args:
            src: Source sequence [batch_size, seq_len, d_model] if batch_first=True
                 or [seq_len, batch_size, d_model] if batch_first=False
            src_mask: Mask for the source sequence (optional)
            src_key_padding_mask: Key padding mask (optional)
            
        Returns:
            Output tensor with the same shape as src
        """
        # Ensure src is in [batch_size, seq_len, d_model] format
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        # Self-attention block with residual connection and layer norm
        src2 = self.norm1(src)
        src2 = self.self_attn(query=src2, key=src2, value=src2, mask=src_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward block with residual connection and layer norm
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        # Return to original format if needed
        if not self.batch_first:
            src = src.transpose(0, 1)
            
        return src


class RelativeTransformerEncoder(nn.Module):
    """TransformerEncoder with Relative Positional Encoding.
    
    This module stacks multiple RelativeTransformerEncoderLayer instances.
    """
    def __init__(self, encoder_layer, num_layers):
        super(RelativeTransformerEncoder, self).__init__()
        
        # Create a ModuleList of encoder layers
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """Forward pass for the relative transformer encoder.
        
        Args:
            src: Source sequence
            mask: Mask for the source sequence (optional)
            src_key_padding_mask: Key padding mask (optional)
            
        Returns:
            Output tensor with the same shape as src
        """
        output = src
        
        # Pass through each encoder layer in sequence
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            
        return output