import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    """Implement the Relative Positional Encoding function.
    This implementation is based on the paper 'Self-Attention with Relative Position Representations'
    by Shaw et al. (2018).
    """
    def __init__(self, d_model=128, dropout=0.2, max_relative_position=20):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_position = max_relative_position
        self.d_model = d_model
        
        # Create relative positional embeddings table
        self.relative_positions_embeddings = nn.Parameter(
            torch.Tensor(2 * max_relative_position + 1, d_model)
        )
        nn.init.xavier_uniform_(self.relative_positions_embeddings)
    
    def _get_relative_positions(self, length):
        """Generate matrix of relative positions between inputs."""
        range_vec = torch.arange(length, device=self.relative_positions_embeddings.device)
        range_mat = range_vec.repeat(length, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift values to be >= 0
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat
    
    def forward(self, x):
        """Forward pass for relative positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Original tensor x (for compatibility with absolute positional encoding)
            and relative position embeddings for use in attention mechanism
        """
        seq_len = x.size(1)
        
        # Get relative position indices for seq_len x seq_len matrix
        relative_positions = self._get_relative_positions(seq_len)
        
        # Get embeddings for each relative position
        rel_embeddings = self.relative_positions_embeddings[relative_positions]
        
        # For compatibility with the original PositionalEncoding class,
        # we return the input tensor unchanged
        return x, self.dropout(rel_embeddings)


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""
    def __init__(self, d_model, num_heads, dropout=0.1, max_relative_position=20):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Relative position encoding
        self.relative_position_encoding = RelativePositionalEncoding(
            d_model=self.d_k,
            dropout=dropout,
            max_relative_position=max_relative_position
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear projections and split into multiple heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scale dot-product attention with relative positional encoding
        Q = Q / self.scale.to(query.device)
        
        # Get relative position embeddings
        _, rel_embeddings = self.relative_position_encoding(query)
        rel_embeddings = rel_embeddings.to(query.device)
        
        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Add relative positional information to attention scores
        rel_scores = self._relative_attention_scores(Q, rel_embeddings, seq_len)
        attention_scores = attention_scores + rel_scores
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output
    
    def _relative_attention_scores(self, Q, rel_embeddings, seq_len):
        """Calculate relative positional attention scores."""
        # Q: [batch_size, num_heads, seq_len, d_k]
        # rel_embeddings: [seq_len, seq_len, d_k]
        
        # Reshape for batch matrix multiplication
        Q_expanded = Q.unsqueeze(3)  # [batch_size, num_heads, seq_len, 1, d_k]
        rel_embeddings_expanded = rel_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len, d_k]
        
        # Calculate relative attention scores
        rel_scores = torch.matmul(
            Q_expanded, 
            rel_embeddings_expanded.transpose(-2, -1)
        ).squeeze(3)  # [batch_size, num_heads, seq_len, seq_len]
        
        return rel_scores