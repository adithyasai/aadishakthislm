"""
Simple test to verify that pruning works with our fixed code.
This version writes output to a log file.
"""
import os
import sys
from pathlib import Path
import logging

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pruning_test_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
logger.info("Setting up Python path...")
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.model import IndicSLM, IndicSLMConfig
from src.pruning import PruningConfig, ModelPruner

logger.info("Starting pruning test...")

# Create a simple model
logger.info("Creating test model...")
config = IndicSLMConfig(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2
)

model = IndicSLM(config)
logger.info("Model created successfully.")

# Count parameters before pruning
total_params = sum(p.numel() for p in model.parameters())
non_zero_before = sum((p != 0).sum().item() for p in model.parameters())
logger.info(f"Total parameters: {total_params}")
logger.info(f"Non-zero parameters before pruning: {non_zero_before}")

# Configure and apply pruning
logger.info("Setting up pruning configuration...")
pruning_config = PruningConfig(
    enabled=True,
    method="magnitude",
    sparsity=0.3,
    schedule="one-shot"
)

logger.info("Creating pruner...")
pruner = ModelPruner(model, pruning_config)

try:
    logger.info("Applying pruning...")
    sparsity = pruner.apply_pruning(epoch=0)
    logger.info(f"Pruning applied successfully. Target sparsity: 0.3, achieved sparsity: {sparsity:.4f}")

    # Count parameters after pruning
    non_zero_after = sum((p != 0).sum().item() for p in model.parameters())
    achieved_sparsity = 1 - non_zero_after / total_params
    logger.info(f"Non-zero parameters after pruning: {non_zero_after}")
    logger.info(f"Parameters pruned: {non_zero_before - non_zero_after}")
    logger.info(f"Actual achieved sparsity: {achieved_sparsity:.4f}")

    # Check if pruning was successful
    if abs(achieved_sparsity - 0.3) < 0.05:
        logger.info("SUCCESS: Pruning test passed!")
    else:
        logger.warning(f"ISSUE: Pruning accuracy not as expected. Target: 0.3, Achieved: {achieved_sparsity:.4f}")

    logger.info("Pruning test completed!")
except Exception as e:
    logger.error(f"Error during pruning: {e}", exc_info=True)
