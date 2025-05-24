import torch
import torch.nn as nn
import math

class RotaryPositionEmbeddings(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) as described in Su et al., 2021
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    RoPE encodes absolute positions with a rotation matrix and naturally incorporates
    explicit relative position dependency in self-attention. This is especially
    useful for Indic languages where long-range dependencies are common.
    """
    def __init__(self, dim, max_position=512, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        
        # Only need d_head/2 frequencies since we apply to pairs of dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _get_cos_sin(self, x, seq_len):
        """
        Calculate cosine and sine values for rotary embeddings
        Args:
            x: Tensor of shape [..., seq_len, dim]
            seq_len: Sequence length
        Returns:
            cos: Tensor of shape [seq_len, dim/2]
            sin: Tensor of shape [seq_len, dim/2]
        """
        # Get position indices
        position = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Calculate position * frequency
        # shape: [seq_len, dim/2]
        sinusoids = torch.einsum("i,j->ij", position, self.inv_freq)
        
        # Calculate cosine and sine values
        # shape: [seq_len, dim/2]
        cos = torch.cos(sinusoids)
        sin = torch.sin(sinusoids)
        
        # Duplicate values to match the original dimension
        # shape: [seq_len, dim]
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        return cos, sin
    
    def _rotate_half(self, x):
        """
        Rotates half the hidden dims of x to prepare for rotation
        Args:
            x: Tensor of shape [..., dim]
        Returns:
            Tensor with half of the dimensions rotated
        """
        # Separate even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Rotate by -90 degrees by swapping and negating
        x_rotated = torch.cat([-x_odd, x_even], dim=-1)
        return x_rotated
    
    def forward(self, query, key):
        """
        Apply rotary position embeddings to query and key tensors
        Args:
            query: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        Returns:
            query_rotated: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key_rotated: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        seq_len = query.size(2)
        
        # Get cos and sin values
        cos, sin = self._get_cos_sin(query, seq_len)
        
        # Reshape for broadcasting
        # [seq_len, dim] -> [1, 1, seq_len, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotary embeddings to query
        query_rotated = (query * cos) + (self._rotate_half(query) * sin)
        
        # Apply rotary embeddings to key
        key_rotated = (key * cos) + (self._rotate_half(key) * sin)
        
        return query_rotated, key_rotated


class RotaryMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings (RoPE).
    This enhances handling of long-range dependencies, particularly important for
    Indic languages with more flexible word order.
    """
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.1,
        max_position=512,
        rope_base=10000.0
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
        
        # Rotary Position Embeddings
        self.rope = RotaryPositionEmbeddings(
            self.d_head, 
            max_position=max_position, 
            base=rope_base
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_head)
    
    def _reshape_for_multihead(self, x):
        """
        Reshape input tensor from [batch_size, seq_len, d_model]
        to [batch_size, num_heads, seq_len, d_head]
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Forward pass for rotary multi-head attention
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
        
        # 3. Apply Rotary Position Embeddings
        q, k = self.rope(q, k)
        
        # 4. Calculate scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 5. Apply masks
        if key_padding_mask is not None:
            # Convert key_padding_mask to attention mask
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, k_len]
            attn_scores = attn_scores.masked_fill(key_padding_mask, float("-inf"))
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Convert 2D mask to 4D mask
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
            attn_scores = attn_scores + attn_mask
        
        # 6. Apply softmax and dropout
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 7. Calculate context vectors
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, q_len, d_head]
        
        # 8. Reshape context vectors
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        # 9. Output projection
        output = self.output_proj(context)
        
        return output, attn_weights