"""
Flash Attention implementation for the Summarization Language Model (SLM).

This module provides an efficient implementation of the Flash Attention mechanism,
which reduces memory usage and improves computation speed for the attention mechanism
in transformer models.

References:
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
  (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
  (Dao et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Union, Any, Dict

logger = logging.getLogger(__name__)

try:
    # Try to import the flash-attn package if available
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTENTION_AVAILABLE = True
    logger.info("Flash Attention package is available and will be used for efficient attention")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.warning(
        "Flash Attention package is not available. "
        "Install it with: pip install flash-attn --no-build-isolation"
    )


class FlashAttention(nn.Module):
    """
    Implementation of Flash Attention for more efficient transformer attention.
    
    This reduces memory usage and improves computation speed for attention mechanisms
    by using a more efficient algorithm that optimizes memory access patterns.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float = 0.0,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        layer_idx: Optional[int] = None,
    ):
        """
        Initialize the Flash Attention module.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            dropout_prob: Attention dropout probability
            causal: Whether to use causal attention (for decoder-only models)
            softmax_scale: Optional scale for attention softmax. If None, uses 1/sqrt(head_dim)
            layer_idx: Optional layer index for logging purposes
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.head_dim = hidden_size // num_attention_heads
        self.causal = causal
        self.layer_idx = layer_idx
        
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        else:
            self.softmax_scale = softmax_scale
        
        if not FLASH_ATTENTION_AVAILABLE:
            # If Flash Attention is not available, log once per layer
            if layer_idx is not None:
                logger.warning(f"Flash Attention not available for layer {layer_idx}, "
                            "falling back to standard attention")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention using Flash Attention if available, otherwise fall back to standard attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads * head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads * head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads * head_dim]
            attention_mask: Optional attention mask of shape [batch_size, 1, 1, seq_len]
            head_mask: Optional mask for specific heads
            
        Returns:
            - Output tensor after attention of shape [batch_size, seq_len, hidden_size]
            - Attention weights (only returned with standard attention)
        """
        batch_size, seq_length = query.shape[0], query.shape[1]
        
        # Reshape query, key, value for multi-head attention
        query = query.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        if FLASH_ATTENTION_AVAILABLE:
            # Use Flash Attention if available
            query = query.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            # Handle padding in attention mask if provided
            if attention_mask is not None:
                # Convert attention mask [batch, 1, 1, seq_len] to [batch, seq_len]
                mask = attention_mask.squeeze(1).squeeze(1).bool()
                
                # Unpad inputs based on attention mask
                seqlens = mask.sum(dim=-1)
                max_seqlen = seqlens.max().item()
                
                if max_seqlen != seq_length:
                    # Handle variable sequence lengths
                    query, indices, cu_seqlens, max_seqlen = unpad_input(query, mask)
                    key, _, _, _ = unpad_input(key, mask)
                    value, _, _, _ = unpad_input(value, mask)
                    
                    # Use variable length flash attention
                    output = flash_attn_varlen_func(
                        query, key, value,
                        cu_seqlens=cu_seqlens, 
                        max_seqlen=max_seqlen,
                        dropout_p=self.dropout_prob if self.training else 0.0,
                        causal=self.causal,
                        scale=self.softmax_scale,
                    )
                    
                    # Pad the output back
                    output = pad_input(output, indices, batch_size, max_seqlen)
                else:
                    # Use standard flash attention for full sequences
                    output = flash_attn_func(
                        query, key, value,
                        dropout_p=self.dropout_prob if self.training else 0.0,
                        causal=self.causal,
                        scale=self.softmax_scale,
                    )
            else:
                # Use standard flash attention when no mask is provided
                output = flash_attn_func(
                    query, key, value,
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    causal=self.causal,
                    scale=self.softmax_scale,
                )
            
            # Restore original shape
            output = output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
            output = output.view(batch_size, seq_length, self.hidden_size)
            
            # Flash Attention doesn't return attention weights to save memory
            return output, None
        else:
            # Fall back to standard attention if Flash Attention is not available
            return self._standard_attention(query, key, value, attention_mask, head_mask)
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute standard scaled dot-product attention as fallback.
        
        Args:
            query, key, value: Input tensors for attention
            attention_mask: Optional mask to prevent attention to padding tokens
            head_mask: Optional mask for specific attention heads
            
        Returns:
            - Output tensor after attention
            - Attention weights
        """
        batch_size, seq_length = query.shape[0], query.shape[1]
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * self.softmax_scale
        
        # Create causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(
                    (seq_length, seq_length),
                    dtype=torch.bool,
                    device=query.device,
                ),
                diagonal=1,
            )
            attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=self.training)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Get context from attention probabilities and values
        context = torch.matmul(attention_probs, value)
        
        # Reshape back to [batch, seq_len, num_heads, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous()
        
        # Combine head outputs
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        return context, attention_probs


def replace_attention_with_flash_attention(
    model: nn.Module,
    causal: bool = True,
    layer_prefix: str = "transformer.layers",
    attention_module_name: str = "attention",
    return_attention_weights: bool = False,
) -> nn.Module:
    """
    Replace standard attention modules in a model with Flash Attention modules.
    
    Args:
        model: The model to modify
        causal: Whether to use causal attention
        layer_prefix: Prefix to locate transformer layers in the model
        attention_module_name: Name of the attention module within each layer
        return_attention_weights: Whether to force returning attention weights
        
    Returns:
        Modified model with Flash Attention (if available)
    """
    if not FLASH_ATTENTION_AVAILABLE:
        logger.warning("Flash Attention is not available, model will use standard attention")
        return model
    
    # Find attention layers in the model
    layer_count = 0
    for name, module in model.named_modules():
        if layer_prefix in name and attention_module_name in name.split(".")[-1]:
            layer_idx = int(name.split(".")[2]) if "." in name else None
            
            # Extract parameters from the old attention module
            if hasattr(module, "num_heads"):
                num_heads = module.num_heads
            elif hasattr(module, "num_attention_heads"):
                num_heads = module.num_attention_heads
            else:
                logger.warning(f"Could not determine number of heads for {name}, skipping")
                continue
            
            if hasattr(module, "hidden_size"):
                hidden_size = module.hidden_size
            elif hasattr(module, "embed_dim"):
                hidden_size = module.embed_dim
            else:
                # Try to infer hidden size from parent
                parent_path = ".".join(name.split(".")[:-1])
                parent_module = model
                for part in parent_path.split("."):
                    if part:
                        parent_module = getattr(parent_module, part)
                
                if hasattr(parent_module, "hidden_size"):
                    hidden_size = parent_module.hidden_size
                else:
                    logger.warning(f"Could not determine hidden size for {name}, skipping")
                    continue
            
            if hasattr(module, "dropout"):
                dropout_prob = module.dropout.p if isinstance(module.dropout, nn.Dropout) else 0.0
            else:
                dropout_prob = 0.1  # Default
            
            # Create Flash Attention module
            flash_attn = FlashAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                dropout_prob=dropout_prob,
                causal=causal,
                layer_idx=layer_idx,
            )
            
            # Override forward method with a wrapper
            original_forward = module.forward
            
            def create_flash_attention_forward(flash_attn, orig_forward):
                def forward_wrapper(self, *args, **kwargs):
                    # Check for different argument patterns
                    if len(args) >= 3:  # Query, key, value passed as positional args
                        query, key, value = args[0], args[1], args[2]
                        attention_mask = kwargs.get("attention_mask", None)
                        head_mask = kwargs.get("head_mask", None)
                        args = args[3:]
                    elif "query" in kwargs and "key" in kwargs and "value" in kwargs:
                        query = kwargs.pop("query")
                        key = kwargs.pop("key")
                        value = kwargs.pop("value")
                        attention_mask = kwargs.pop("attention_mask", None)
                        head_mask = kwargs.pop("head_mask", None)
                    elif "hidden_states" in kwargs:  # For some HuggingFace models
                        hidden_states = kwargs["hidden_states"]
                        query = key = value = hidden_states
                        attention_mask = kwargs.get("attention_mask", None)
                        head_mask = kwargs.get("head_mask", None)
                    else:
                        # If args don't match expected pattern, fall back to original implementation
                        logger.warning(f"Couldn't match args for Flash Attention, using standard attention")
                        return orig_forward(*args, **kwargs)
                    
                    # Call Flash Attention
                    outputs, attention_weights = flash_attn(
                        query, key, value, 
                        attention_mask=attention_mask,
                        head_mask=head_mask
                    )
                    
                    if return_attention_weights:
                        # If the original expects attention weights, we need to provide them
                        return outputs, attention_weights
                    else:
                        # Otherwise just return outputs
                        return outputs
                
                return forward_wrapper
            
            # Apply the wrapper
            module.forward = create_flash_attention_forward(flash_attn, original_forward).__get__(module)
            layer_count += 1
    
    logger.info(f"Replaced attention with Flash Attention in {layer_count} layers")
    return model


def apply_flash_attention(model: nn.Module) -> nn.Module:
    """
    Apply Flash Attention to a model, automatically detecting model type.
    
    Args:
        model: The transformer model to modify
        
    Returns:
        Model with Flash Attention (if available)
    """
    if not FLASH_ATTENTION_AVAILABLE:
        return model
    
    # Try to detect model type and use appropriate parameters
    model_name = model.__class__.__name__
    
    if "Indic" in model_name and "SLM" in model_name:
        # Our custom model
        return replace_attention_with_flash_attention(
            model,
            causal=True,  # Assuming our SLM uses causal attention
            layer_prefix="transformer.layers",
            attention_module_name="attention",
        )
    elif "GPT" in model_name or "LLaMA" in model_name or "Llama" in model_name:
        # GPT/LLaMA style models
        return replace_attention_with_flash_attention(
            model, 
            causal=True,
            layer_prefix="transformer.h",
            attention_module_name="attn",
        )
    elif "BERT" in model_name or "RoBERTa" in model_name:
        # BERT style models
        return replace_attention_with_flash_attention(
            model,
            causal=False,
            layer_prefix="encoder.layer",
            attention_module_name="attention",
        )
    else:
        # Generic approach - may not work for all models
        logger.warning(f"Unknown model type: {model_name}. "
                    f"Attempting to apply Flash Attention with default settings.")
        return replace_attention_with_flash_attention(model, causal=True)
