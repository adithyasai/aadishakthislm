"""
Model pruning support for Summarization Language Model (SLM).

This module provides functions for pruning the SLM model to reduce model size
and potentially improve inference speed. Supports different pruning strategies:
1. Magnitude-based pruning (one-shot)
2. Gradual pruning during training
3. Structured pruning (optional)

References:
- "To prune, or not to prune: exploring the efficacy of pruning for model compression" (Zhu & Gupta, 2017)
- "The State of Sparsity in Deep Neural Networks" (Gale et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
import math
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
from pathlib import Path
import copy
import numpy as np

logger = logging.getLogger(__name__)

class PruningConfig:
    """Configuration for model pruning."""
    
    def __init__(
        self,
        enabled: bool = False,
        method: str = "magnitude",  # "magnitude", "random", "structured"
        sparsity: float = 0.3,      # Target sparsity (0.0 to 1.0)
        schedule: str = "one-shot",  # "one-shot", "gradual"
        start_epoch: int = 0,        # For gradual pruning
        end_epoch: int = 10,         # For gradual pruning
        frequency: int = 1,          # Pruning frequency (epochs)
        structured_dim: int = 0,     # For structured pruning (0: unstructured)
        exclude_layers: List[str] = None,  # Layer names to exclude from pruning
    ):
        """
        Initialize pruning configuration.
        
        Args:
            enabled: Whether pruning is enabled
            method: Pruning method ("magnitude", "random", "structured")
            sparsity: Target sparsity ratio (0.0 to 1.0)
            schedule: Pruning schedule ("one-shot", "gradual")
            start_epoch: Starting epoch for gradual pruning
            end_epoch: Ending epoch for gradual pruning
            frequency: Pruning frequency in epochs
            structured_dim: Dimension for structured pruning (0: unstructured)
            exclude_layers: Layer names to exclude from pruning
        """
        self.enabled = enabled
        self.method = method
        self.sparsity = sparsity
        self.schedule = schedule
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.frequency = frequency
        self.structured_dim = structured_dim
        self.exclude_layers = exclude_layers or []
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {self.sparsity}")
        
        if self.method not in ["magnitude", "random", "structured"]:
            raise ValueError(f"Unknown pruning method: {self.method}")
        
        if self.schedule not in ["one-shot", "gradual"]:
            raise ValueError(f"Unknown pruning schedule: {self.schedule}")
        
        if self.schedule == "gradual":
            if self.start_epoch >= self.end_epoch:
                raise ValueError(f"start_epoch ({self.start_epoch}) must be less than end_epoch ({self.end_epoch})")


class ModelPruner:
    """Handle model pruning operations."""
    
    def __init__(
        self,
        model: nn.Module,
        config: PruningConfig
    ):
        """
        Initialize model pruner.
        
        Args:
            model: The model to prune
            config: Pruning configuration
        """
        self.model = model
        self.config = config
        self.original_model = None  # For storing the original model state
        self.pruned_weights = {}    # To track which weights have been pruned
        self.current_sparsity = 0.0
        
        # Store initial model if pruning is enabled
        if config.enabled:
            self.original_model = copy.deepcopy(model.state_dict())
    
    def compute_sparsity(self, epoch: int) -> float:
        """
        Compute target sparsity for the current epoch based on schedule.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Target sparsity for this epoch
        """
        if self.config.schedule == "one-shot":
            # One-shot pruning applies full sparsity immediately
            return self.config.sparsity
        elif self.config.schedule == "gradual":
            # Gradual pruning increases sparsity over time
            if epoch < self.config.start_epoch:
                return 0.0
            elif epoch >= self.config.end_epoch:
                return self.config.sparsity
            else:
                # Linear schedule from start_epoch to end_epoch
                pruning_progress = (epoch - self.config.start_epoch) / (self.config.end_epoch - self.config.start_epoch)
                # Cubic schedule for gradual pruning (following Zhu & Gupta, 2017)
                scaled_progress = pruning_progress ** 3  # Cubic growth
                return self.config.sparsity * scaled_progress
    
    def should_prune_on_epoch(self, epoch: int) -> bool:
        """
        Determine if pruning should be applied on the current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            True if pruning should be applied, False otherwise
        """
        if not self.config.enabled:
            return False
            
        if self.config.schedule == "one-shot":
            return epoch == self.config.start_epoch
        elif self.config.schedule == "gradual":
            if epoch < self.config.start_epoch or epoch > self.config.end_epoch:
                return False
            return (epoch - self.config.start_epoch) % self.config.frequency == 0
        
        return False
    
    def apply_pruning(self, epoch: int) -> float:
        """
        Apply pruning to the model based on configuration.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current sparsity level after pruning
        """
        if not self.config.enabled:
            logger.info("Pruning is disabled. Skipping.")
            return 0.0
            
        if not self.should_prune_on_epoch(epoch):
            return self.current_sparsity
            
        # Compute sparsity for this epoch
        target_sparsity = self.compute_sparsity(epoch)
        logger.info(f"Applying pruning at epoch {epoch} with target sparsity {target_sparsity:.4f}")
        
        # Determine which parameters to prune
        parameters_to_prune = []
          # Collect prunable parameters
        for name, module in self.model.named_modules():
            # Skip excluded layers
            if any(excluded in name for excluded in self.config.exclude_layers):
                continue
                
            # Currently support pruning Linear and Conv2d layers
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
          # Remove any previous pruning masks with better error handling
        for module, param_name in parameters_to_prune:
            # Check if the parameter has been pruned before removing
            if hasattr(module, param_name + '_mask') and hasattr(module, param_name + '_orig'):
                try:
                    prune.remove(module, param_name)
                except Exception as e:
                    logger.warning(f"Could not remove pruning from {module}. Error: {e}")
                    # Continue with pruning anyway
        
        # Apply pruning based on method
        if self.config.method == "magnitude":
            # Global magnitude pruning (most effective)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=target_sparsity,
            )
        elif self.config.method == "random":
            # Random pruning (baseline)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=target_sparsity,
            )
        elif self.config.method == "structured" and self.config.structured_dim > 0:
            # Structured pruning - prune entire channels/neurons
            for module, param_name in parameters_to_prune:
                if isinstance(module, nn.Linear):
                    prune.ln_structured(
                        module, 
                        param_name, 
                        amount=target_sparsity, 
                        n=2,  # L2 norm for structured pruning
                        dim=self.config.structured_dim
                    )
                elif isinstance(module, nn.Conv2d):
                    prune.ln_structured(
                        module, 
                        param_name, 
                        amount=target_sparsity, 
                        n=2, 
                        dim=self.config.structured_dim
                    )
        
        # Track which weights have been pruned
        for module, param_name in parameters_to_prune:
            param_name_mask = param_name + "_mask"
            if hasattr(module, param_name_mask):
                mask = getattr(module, param_name_mask)
                key = f"{id(module)}.{param_name}"
                self.pruned_weights[key] = mask
        
        # Calculate and return actual sparsity
        total_params = 0
        zero_params = 0
        
        for module, param_name in parameters_to_prune:
            param = getattr(module, param_name)
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        if total_params > 0:
            self.current_sparsity = zero_params / total_params
            logger.info(f"Applied pruning: {zero_params} of {total_params} weights pruned "
                      f"({self.current_sparsity:.4f} sparsity)")
        else:
            logger.warning("No parameters were pruned.")
            self.current_sparsity = 0.0
        
        return self.current_sparsity
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about model pruning.
        
        Returns:
            Dictionary with pruning statistics
        """
        statistics = {
            "enabled": self.config.enabled,
            "method": self.config.method,
            "target_sparsity": self.config.sparsity,
            "current_sparsity": self.current_sparsity,
            "schedule": self.config.schedule,
            "layer_sparsity": {}
        }
        
        # Compute sparsity for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight
                total = weight.numel()
                zero_count = (weight == 0).sum().item()
                sparsity = zero_count / total if total > 0 else 0
                statistics["layer_sparsity"][name] = {
                    "total_params": total,
                    "zero_params": zero_count,
                    "sparsity": sparsity
                }
        
        return statistics
    
    def restore_original_weights(self):
        """Restore the model to its original state before pruning."""
        if self.original_model is not None:
            self.model.load_state_dict(self.original_model)
            logger.info("Restored original model weights (before pruning)")
        else:
            logger.warning("No original weights stored, cannot restore")


def apply_magnitude_pruning(model: nn.Module, sparsity: float = 0.3):
    """
    Apply one-shot magnitude-based pruning to the model.
    
    Args:
        model: The model to prune
        sparsity: Target sparsity ratio (0.0 to 1.0)
        
    Returns:
        Pruned model
    """
    logger.info(f"Applying magnitude pruning with {sparsity:.2f} sparsity")
    
    # Create pruning configuration
    config = PruningConfig(
        enabled=True,
        method="magnitude",
        sparsity=sparsity,
        schedule="one-shot"
    )
    
    # Create pruner and apply pruning
    pruner = ModelPruner(model, config)
    pruner.apply_pruning(epoch=0)
    
    # Return pruned model
    return model


def save_pruned_model(model: nn.Module, save_dir: str):
    """
    Save a pruned model with its pruning masks.
    
    Args:
        model: The pruned model
        save_dir: Directory to save the model
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(save_dir, "pruned_model.pt"))
    
    # Save pruning information
    pruning_info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Check if this module has been pruned
            if hasattr(module, "weight_mask"):
                mask = module.weight_mask
                pruning_info[f"{name}.weight"] = {
                    "sparsity": (mask == 0).sum().item() / mask.numel(),
                    "mask_shape": list(mask.shape)
                }
    
    torch.save(pruning_info, os.path.join(save_dir, "pruning_info.pt"))
    logger.info(f"Saved pruned model to {save_dir}")


def load_pruned_model(model_class, model_config, load_dir: str):
    """
    Load a pruned model with its pruning masks.
    
    Args:
        model_class: The model class to instantiate
        model_config: Configuration for model initialization
        load_dir: Directory containing the saved model
        
    Returns:
        Loaded pruned model
    """
    import os
    
    # Create model instance
    model = model_class(model_config)
    
    # Load state dict
    model.load_state_dict(torch.load(os.path.join(load_dir, "pruned_model.pt")))
    
    # Load pruning information (for reference only, masks are part of state_dict)
    pruning_info = torch.load(os.path.join(load_dir, "pruning_info.pt"))
    
    logger.info(f"Loaded pruned model from {load_dir}")
    logger.info(f"Model has pruning masks for {len(pruning_info)} parameters")
    
    return model


def evaluate_pruning_impact(
    original_model: nn.Module, 
    pruned_model: nn.Module, 
    test_inputs: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """
    Evaluate the impact of pruning on model performance.
    
    Args:
        original_model: Original unpruned model
        pruned_model: Pruned model
        test_inputs: Test inputs for the model
        
    Returns:
        Dictionary with performance metrics
    """
    original_model.eval()
    pruned_model.eval()
    
    results = {}
    
    # Measure inference time
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    # Warm-up runs
    for _ in range(5):
        with torch.no_grad():
            _ = original_model(**test_inputs)
            _ = pruned_model(**test_inputs)
    
    # Measure original model
    if start_time:
        start_time.record()
    
    with torch.no_grad():
        original_output = original_model(**test_inputs)
    
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        original_time = start_time.elapsed_time(end_time)
    else:
        import time
        start = time.time()
        with torch.no_grad():
            _ = original_model(**test_inputs)
        original_time = (time.time() - start) * 1000  # Convert to ms
    
    # Measure pruned model
    if start_time:
        start_time.record()
    
    with torch.no_grad():
        pruned_output = pruned_model(**test_inputs)
    
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        pruned_time = start_time.elapsed_time(end_time)
    else:
        import time
        start = time.time()
        with torch.no_grad():
            _ = pruned_model(**test_inputs)
        pruned_time = (time.time() - start) * 1000  # Convert to ms
    
    # Calculate output difference
    original_logits = original_output["logits"]
    pruned_logits = pruned_output["logits"]
    
    output_diff = {
        "mse": torch.nn.functional.mse_loss(pruned_logits, original_logits).item(),
        "mae": torch.nn.functional.l1_loss(pruned_logits, original_logits).item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            pruned_logits.view(-1), original_logits.view(-1), dim=0
        ).item()
    }
    
    # Count parameters
    def count_non_zero_params(model: nn.Module):
        total = 0
        non_zero = 0
        for param in model.parameters():
            total += param.numel()
            non_zero += (param != 0).sum().item()
        return total, non_zero
    
    orig_total, orig_non_zero = count_non_zero_params(original_model)
    pruned_total, pruned_non_zero = count_non_zero_params(pruned_model)
    
    # Compile results
    results = {
        "inference_time": {
            "original_ms": original_time,
            "pruned_ms": pruned_time,
            "speedup": original_time / pruned_time if pruned_time > 0 else 0
        },
        "parameters": {
            "original_total": orig_total,
            "original_non_zero": orig_non_zero,
            "pruned_non_zero": pruned_non_zero,
            "sparsity": 1 - (pruned_non_zero / orig_total) if orig_total > 0 else 0
        },
        "output_difference": output_diff
    }
    
    return results
