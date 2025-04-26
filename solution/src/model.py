import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Dict

class IndicSLMConfig(PretrainedConfig):
    model_type = "indic_slm"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
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
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
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
    ) -> Tuple[torch.Tensor, ...]:
        
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
        
        # Encode
        encoder_outputs = embeddings
        for encoder_layer in self.encoder:
            encoder_outputs = encoder_layer(encoder_outputs, attention_mask)
        
        # Decode
        decoder_outputs = encoder_outputs
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
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Self attention
        attended = self.attention(x, x, x, key_padding_mask=attention_mask.eq(0))[0]
        x = self.layernorm1(x + self.dropout(attended))
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: IndicSLMConfig):
        super().__init__()
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
    
    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Self attention
        self_attended = self.self_attention(x, x, x, key_padding_mask=attention_mask.eq(0))[0]
        x = self.layernorm1(x + self.dropout(self_attended))
        
        # Cross attention with encoder outputs
        cross_attended = self.cross_attention(x, encoder_outputs, encoder_outputs, key_padding_mask=attention_mask.eq(0))[0]
        x = self.layernorm2(x + self.dropout(cross_attended))
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layernorm3(x + self.dropout(ff_output))
        
        return x