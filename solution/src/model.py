import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Dict, Union, List
from src.positional_encoding import RelativeMultiHeadAttention
from src.rope import RotaryMultiHeadAttention
from src.language_adapters import LanguageAdapterCollection, AdapterEncoderLayer, AdapterDecoderLayer, detect_language

class IndicSLMConfig(PretrainedConfig):
    model_type = "indic_slm"
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        attention_type: str = "default",  # Options: "default", "relative", "rotary"
        max_relative_position: int = 32,  # For relative positions
        rope_base: float = 10000.0,  # For rotary embeddings
        use_adapters: bool = False,  # Whether to use language adapters
        adapter_size: int = 64,      # Size of adapter bottleneck
        languages: List[str] = None, # Languages to create adapters for
        quantization: Dict = None,   # Quantization settings
        pruning: Dict = None,        # Pruning settings
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.attention_type = attention_type
        self.max_relative_position = max_relative_position
        self.rope_base = rope_base
        self.use_adapters = use_adapters
        self.adapter_size = adapter_size
        self.languages = languages or ["hi", "te"]  # Default to Hindi and Telugu
        self.quantization = quantization or {"enabled": False, "bits": 8, "type": "dynamic"}
        self.pruning = pruning or {"enabled": False, "sparsity": 0.3, "method": "magnitude"}

class IndicSLMPreTrainedModel(PreTrainedModel):
    config_class = IndicSLMConfig
    base_model_prefix = "indic_slm"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class IndicSLM(IndicSLMPreTrainedModel):
    def __init__(self, config: IndicSLMConfig):
        super().__init__(config)
        
        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, config.hidden_size),
            'token_type_embeddings': nn.Embedding(2, config.hidden_size)  # For language identification
        })
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Create base encoder and decoder layers
        encoder_layers = [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        decoder_layers = [TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        
        # If language adapters are enabled, wrap the base layers with adapter layers
        if config.use_adapters:
            # Create language adapter collection
            self.language_adapters = LanguageAdapterCollection(
                hidden_size=config.hidden_size,
                languages=config.languages,
                adapter_size=config.adapter_size
            )
            
            # Wrap encoder layers with adapter layers
            self.encoder = nn.ModuleList([
                AdapterEncoderLayer(encoder_layer, self.language_adapters) 
                for encoder_layer in encoder_layers
            ])
            
            # Wrap decoder layers with adapter layers
            self.decoder = nn.ModuleList([
                AdapterDecoderLayer(decoder_layer, self.language_adapters) 
                for decoder_layer in decoder_layers
            ])
        else:
            # Use base layers without adapters
            self.encoder = nn.ModuleList(encoder_layers)
            self.decoder = nn.ModuleList(decoder_layers)
        
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        language_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids) if token_type_ids is not None else 0
        
        # Combine embeddings
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Try to detect language if not provided and adapters are enabled
        if language_id is None and hasattr(self, 'language_adapters') and input_ids.size(0) == 1:
            # We can only reliably detect language for a single sequence
            # Get the input text from the tokenizer's decode function if available
            # This is just a fallback and would require access to the tokenizer
            if hasattr(self, 'tokenizer'):
                sample_text = self.tokenizer.decode(input_ids[0])
                language_id = detect_language(sample_text)
        
        # Encode with language adaptation if applicable
        encoder_outputs = embeddings
        if hasattr(self, 'language_adapters') and language_id is not None:
            # Pass language_id to each adapter-enabled encoder layer
            for encoder_layer in self.encoder:
                encoder_outputs = encoder_layer(encoder_outputs, attention_mask, language_id)
        else:
            # Standard forward pass without language adaptation
            for encoder_layer in self.encoder:
                encoder_outputs = encoder_layer(encoder_outputs, attention_mask)
        
        # Decode with language adaptation if applicable
        decoder_outputs = encoder_outputs
        if hasattr(self, 'language_adapters') and language_id is not None:
            # Pass language_id to each adapter-enabled decoder layer
            for decoder_layer in self.decoder:
                decoder_outputs = decoder_layer(decoder_outputs, encoder_outputs, attention_mask, language_id)
        else:
            # Standard forward pass without language adaptation
            for decoder_layer in self.decoder:
                decoder_outputs = decoder_layer(decoder_outputs, encoder_outputs, attention_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_outputs)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: IndicSLMConfig):
        super().__init__()
        
        # Select attention mechanism based on configuration
        if config.attention_type == "relative":
            self.attention = RelativeMultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                max_relative_position=config.max_relative_position
            )
        elif config.attention_type == "rotary":
            self.attention = RotaryMultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                max_position=config.max_position_embeddings,
                rope_base=config.rope_base
            )
        else:  # default
            self.attention = nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
            
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_type = config.attention_type
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Self attention
        if self.attention_type == "default":
            attended = self.attention(x, x, x, key_padding_mask=attention_mask.eq(0))[0]
        else:  # relative or rotary
            attended, _ = self.attention(x, x, x, key_padding_mask=attention_mask.eq(0))
            
        x = self.layernorm1(x + self.dropout(attended))
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: IndicSLMConfig):
        super().__init__()
        
        # Select self-attention mechanism based on configuration
        if config.attention_type == "relative":
            self.self_attention = RelativeMultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                max_relative_position=config.max_relative_position
            )
            self.cross_attention = RelativeMultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                max_relative_position=config.max_relative_position
            )
        elif config.attention_type == "rotary":
            self.self_attention = RotaryMultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                max_position=config.max_position_embeddings,
                rope_base=config.rope_base
            )
            self.cross_attention = RotaryMultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                max_position=config.max_position_embeddings,
                rope_base=config.rope_base
            )
        else:  # default
            self.self_attention = nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
            self.cross_attention = nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
            
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_type = config.attention_type
    
    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Self attention
        if self.attention_type == "default":
            self_attended = self.self_attention(x, x, x, key_padding_mask=attention_mask.eq(0))[0]
        else:  # relative or rotary
            self_attended, _ = self.self_attention(x, x, x, key_padding_mask=attention_mask.eq(0))
            
        x = self.layernorm1(x + self.dropout(self_attended))
        
        # Cross attention with encoder outputs
        if self.attention_type == "default":
            cross_attended = self.cross_attention(x, encoder_outputs, encoder_outputs, key_padding_mask=attention_mask.eq(0))[0]
        else:  # relative or rotary
            cross_attended, _ = self.cross_attention(x, encoder_outputs, encoder_outputs, key_padding_mask=attention_mask.eq(0))
            
        x = self.layernorm2(x + self.dropout(cross_attended))
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layernorm3(x + self.dropout(ff_output))
        
        return x


def quantize_model(model, quantization_config=None, calibration_data=None):
    """
    Quantize the model using the specified quantization configuration.
    
    Args:
        model: The model to quantize
        quantization_config: Configuration for quantization
        calibration_data: Data for calibrating static quantization (if applicable)
        
    Returns:
        Quantized model wrapper
    """
    # Import quantization utilities here to avoid circular imports
    try:
        # First try to import from the regular quantization module
        from src.quantization import (
            apply_weight_only_quantization,
            QuantizedModelWrapper
        )
    except ImportError:
        # Fall back to the simplified version
        from src.simple_quantization import (
            apply_weight_only_quantization,
            QuantizedModelWrapper
        )
    
    # Use model's config quantization settings if not provided
    if quantization_config is None:
        if hasattr(model.config, 'quantization'):
            quantization_config = model.config.quantization
        else:
            quantization_config = {"enabled": False, "bits": 8, "type": "weight_only"}
    
    # Skip if quantization is not enabled
    if not quantization_config.get("enabled", False):
        wrapper = QuantizedModelWrapper(model, quantization_config)
        wrapper.is_quantized = False
        return wrapper
    
    # Get quantization parameters
    bits = quantization_config.get("bits", 8)
    quant_type = quantization_config.get("type", "weight_only")
    
    # For reliability, always use weight-only quantization
    return apply_weight_only_quantization(model, bits=bits)