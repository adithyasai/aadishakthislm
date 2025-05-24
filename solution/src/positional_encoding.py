import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePositionalEncoding(nn.Module):
    """
    Implements relative positional encodings as described in Shaw et al., 2018
    "Self-Attention with Relative Position Representations"
    
    This is particularly beneficial for languages with flexible word order like Hindi and Telugu.
    """
    def __init__(self, d_model, max_relative_position=32):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.d_model = d_model
        
        # Create relative position embeddings for keys and values
        self.relative_key_embedding = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
        self.relative_value_embedding = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
    
    def _get_relative_positions(self, length):
        """
        Compute relative position indices for a sequence of given length
        Returns a tensor of shape [length, length]
        """
        # Create position indices for each token
        range_vec = torch.arange(length, device=self.relative_key_embedding.weight.device)
        # For each position i, calculate the relative position i-j for all j
        relative_positions = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        
        # Clip the relative positions to the maximum relative position
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to make indices non-negative for embedding lookup
        return relative_positions + self.max_relative_position
    
    def forward(self, length):
        """
        Generate relative position embeddings for a sequence of given length
        Returns:
            relative_keys: [length, length, d_model]
            relative_values: [length, length, d_model]
        """
        # Get relative position indices
        relative_positions = self._get_relative_positions(length)
        
        # Look up relative position embeddings for keys and values
        relative_keys = self.relative_key_embedding(relative_positions)
        relative_values = self.relative_value_embedding(relative_positions)
        
        return relative_keys, relative_values


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encodings.
    This enhances the standard attention mechanism to better capture
    word order relationships in Indic languages.
    """
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.1,
        max_relative_position=32
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Linear projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Simplified relative positional bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_relative_position + 1))
        self.max_relative_position = max_relative_position
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_head)
    
    def _get_relative_positions(self, length):
        """
        Compute relative position indices for a sequence of given length
        Returns a tensor of shape [length, length]
        """
        # Create position indices for each token
        range_vec = torch.arange(length, device=self.rel_pos_bias.device)
        # For each position i, calculate the relative position i-j for all j
        relative_positions = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        
        # Clip the relative positions to the maximum relative position
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to make indices non-negative for bias lookup
        return relative_positions + self.max_relative_position
    
    def _reshape_for_multihead(self, x):
        """
        Reshape input tensor from [batch_size, seq_len, d_model]
        to [batch_size, num_heads, seq_len, d_head]
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Forward pass for relative multi-head attention
        Args:
            query: [batch_size, q_len, d_model]
            key: [batch_size, k_len, d_model]
            value: [batch_size, k_len, d_model]
            key_padding_mask: [batch_size, k_len] mask for padded keys
            attn_mask: [q_len, k_len] or [batch_size, num_heads, q_len, k_len]
                       causal mask for autoregressive language modeling
        Returns:
            output: [batch_size, q_len, d_model]
            attention_weights: [batch_size, num_heads, q_len, k_len]
        """
        batch_size, q_len, _ = query.size()
        k_len = key.size(1)
        
        # 1. Linear projection
        q = self.query_proj(query)  # [batch_size, q_len, d_model]
        k = self.key_proj(key)      # [batch_size, k_len, d_model]
        v = self.value_proj(value)  # [batch_size, k_len, d_model]
        
        # 2. Reshape for multi-head attention
        q = self._reshape_for_multihead(q)  # [batch_size, num_heads, q_len, d_head]
        k = self._reshape_for_multihead(k)  # [batch_size, num_heads, k_len, d_head]
        v = self._reshape_for_multihead(v)  # [batch_size, num_heads, k_len, d_head]
        
        # 3. Calculate attention scores
        # Standard dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, q_len, k_len]
        
        # Add relative position bias
        rel_pos_indices = self._get_relative_positions(max(q_len, k_len))
        rel_pos_indices = rel_pos_indices[:q_len, :k_len]
        rel_pos_bias = self.rel_pos_bias[rel_pos_indices]
        
        # Add the relative position bias to the attention scores
        scores = scores + rel_pos_bias.unsqueeze(0).unsqueeze(0)
        
        # 4. Apply masks
        if key_padding_mask is not None:
            # Convert key_padding_mask to attention mask
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, k_len]
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Convert 2D mask to 4D mask
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
            scores = scores + attn_mask
        
        # 5. Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, q_len, k_len]
        attn_weights = self.dropout(attn_weights)
        
        # 6. Calculate context vectors
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, q_len, d_head]
        
        # 7. Reshape context vectors
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        # 8. Output projection
        output = self.output_proj(context)
        
        return output, attn_weights