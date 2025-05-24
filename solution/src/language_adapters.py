import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class LanguageAdapter(nn.Module):
    """
    A lightweight adapter module for language-specific customization.
    
    Adapters are small modules inserted between layers of a pre-trained model 
    to adapt it to specific languages without full fine-tuning.
    """
    def __init__(
        self, 
        hidden_size: int, 
        adapter_size: int = 64,
        adapter_initializer_range: float = 0.02,
        adapter_act: str = "gelu"
    ):
        """
        Initialize a language adapter module.
        
        Args:
            hidden_size: Size of the hidden states in the main model
            adapter_size: Bottleneck dimension for the adapter
            adapter_initializer_range: Initialization range for adapter weights
            adapter_act: Activation function to use in adapter
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.adapter_initializer_range = adapter_initializer_range
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # Up projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Activation function
        if adapter_act == "gelu":
            self.activation = F.gelu
        elif adapter_act == "relu":
            self.activation = F.relu
        elif adapter_act == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {adapter_act}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the adapter module"""
        nn.init.normal_(self.down_proj.weight, std=self.adapter_initializer_range)
        nn.init.normal_(self.up_proj.weight, std=self.adapter_initializer_range)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Adapted hidden states
        """
        # Apply adapter layers (residual connection is important)
        residual = hidden_states
        
        # Down projection
        hidden_states = self.down_proj(hidden_states)
        
        # Activation
        hidden_states = self.activation(hidden_states)
        
        # Up projection
        hidden_states = self.up_proj(hidden_states)
        
        # Add residual connection
        output = residual + hidden_states
        
        return output


class LanguageAdapterCollection:
    """
    Manages a collection of language adapters for multiple languages.
    """
    def __init__(self, hidden_size: int, languages: List[str], adapter_size: int = 64):
        """
        Initialize a collection of language adapters.
        
        Args:
            hidden_size: Size of hidden states in the main model
            languages: List of language codes to create adapters for
            adapter_size: Size of adapter bottleneck dimension
        """
        self.hidden_size = hidden_size
        self.languages = languages
        self.adapter_size = adapter_size
        
        # Create a dictionary of language adapters
        self.adapters = nn.ModuleDict({
            lang: LanguageAdapter(hidden_size, adapter_size) 
            for lang in languages
        })
        
        logger.info(f"Created language adapters for: {', '.join(languages)}")
    
    def __getitem__(self, lang: str) -> LanguageAdapter:
        """Get adapter for a specific language"""
        if lang not in self.adapters:
            logger.warning(f"No adapter for language '{lang}'. Using first available adapter.")
            # Default to first language if not found
            return self.adapters[next(iter(self.adapters))]
        return self.adapters[lang]


class AdapterEncoderLayer(nn.Module):
    """
    Encoder layer enhanced with language adapters.
    This wraps around the regular encoder layer and adds language adaptation.
    """
    def __init__(self, encoder_layer, language_adapters: LanguageAdapterCollection):
        """
        Initialize an adapter-enhanced encoder layer.
        
        Args:
            encoder_layer: The original encoder layer to wrap
            language_adapters: Collection of language adapters
        """
        super().__init__()
        self.encoder_layer = encoder_layer
        self.language_adapters = language_adapters
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: torch.Tensor,
                language_id: str = None) -> torch.Tensor:
        """
        Forward pass with language adaptation.
        
        Args:
            x: Input tensor
            attention_mask: Attention mask
            language_id: Language identifier to select appropriate adapter
            
        Returns:
            torch.Tensor: Adapted output
        """
        # Run through base encoder layer
        output = self.encoder_layer(x, attention_mask)
        
        # Apply language adapter if language_id is provided
        if language_id is not None and hasattr(self, 'language_adapters'):
            adapter = self.language_adapters[language_id]
            output = adapter(output)
        
        return output


class AdapterDecoderLayer(nn.Module):
    """
    Decoder layer enhanced with language adapters.
    This wraps around the regular decoder layer and adds language adaptation.
    """
    def __init__(self, decoder_layer, language_adapters: LanguageAdapterCollection):
        """
        Initialize an adapter-enhanced decoder layer.
        
        Args:
            decoder_layer: The original decoder layer to wrap
            language_adapters: Collection of language adapters
        """
        super().__init__()
        self.decoder_layer = decoder_layer
        self.language_adapters = language_adapters
    
    def forward(self, 
                x: torch.Tensor, 
                encoder_outputs: torch.Tensor, 
                attention_mask: torch.Tensor,
                language_id: str = None) -> torch.Tensor:
        """
        Forward pass with language adaptation.
        
        Args:
            x: Input tensor
            encoder_outputs: Outputs from the encoder
            attention_mask: Attention mask
            language_id: Language identifier to select appropriate adapter
            
        Returns:
            torch.Tensor: Adapted output
        """
        # Run through base decoder layer
        output = self.decoder_layer(x, encoder_outputs, attention_mask)
        
        # Apply language adapter if language_id is provided
        if language_id is not None and hasattr(self, 'language_adapters'):
            adapter = self.language_adapters[language_id]
            output = adapter(output)
        
        return output


def detect_language(text: str) -> str:
    """
    Detect language from text.
    This is a simple implementation that looks for language markers or script patterns.
    
    Args:
        text: Input text
        
    Returns:
        str: Detected language code ('hi', 'te', or None if unknown)
    """
    # Check for explicit language markers like <hi> or <te>
    if "<hi>" in text.lower():
        return "hi"
    elif "<te>" in text.lower():
        return "te"
    
    # Simple script-based detection
    # Check for Devanagari characters (Hindi)
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    
    # Check for Telugu characters
    telugu_count = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    
    # Make a decision based on character counts
    if devanagari_count > telugu_count:
        return "hi"
    elif telugu_count > devanagari_count:
        return "te"
    
    # Default to None if we can't determine
    return None