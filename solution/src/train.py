import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import get_linear_schedule_with_warmup
import wandb
from typing import Dict, Any, Optional, Iterator
import os
import json
import argparse
from pathlib import Path
import gc
import psutil
import logging
from tqdm import tqdm
import sys

# Set WandB API key
os.environ["WANDB_API_KEY"] = "312be93beceeacf746205bed4dd96f155f068cf1"

# Setup base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "models"

# Add parent directory to Python path to ensure imports work
import sys
sys.path.append(str(BASE_DIR))

from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer
from src.data_processor import DataProcessor
from src.pruning import PruningConfig, ModelPruner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitorCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        if memory_gb > 8:  # Lower threshold to 8GB
            logger.warning(f"High memory usage detected: {memory_gb:.2f}GB")
            gc.collect()
            torch.cuda.empty_cache()

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, lang="te", max_length=128):
        self.tokenizer = tokenizer
        self.lang = lang
        self.max_length = max_length
        
        # Simple sample data
        self.samples = []
        
        # Add sample for Telugu
        if lang == "te":
            sample_text = """తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష. ఇది భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో నాలుగవది మరియు ద్రావిడ భాషా కుటుంబంలో రెండవ అతిపెద్ద భాష."""
        else:  # Hindi
            sample_text = """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत गणराज्य की आधिकारिक भाषाओं में से एक है।"""
        
        for i in range(50):  # Create 50 samples for training
            self.samples.append(f"<{lang}> {sample_text}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Encode with special tokens
        encoding = self.tokenizer.encode(text)
        
        # Truncate or pad
        input_ids = encoding["input_ids"][:self.max_length]
        attention_mask = encoding["attention_mask"][:self.max_length]
        
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # Create labels for autoregressive training (shifted by 1)
        labels = input_ids[1:] + [self.tokenizer.pad_token_id]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

class IndicSLMTrainer(pl.LightningModule):
    def __init__(self, config_path: str = str(CONFIG_PATH), pruning_config: Optional[Dict] = None):
        super().__init__()
        self.save_hyperparameters()
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Initialize model with smaller configuration
        self.config['n_embd'] = 192  # Reduced embedding size
        self.config['n_layer'] = 3   # Reduced number of layers
        self.config['n_head'] = 6    # Reduced number of heads
        
        # Initialize pruning if provided
        self.pruning_config = pruning_config
        
        self.model_config = IndicSLMConfig(
            vocab_size=self.config['vocab_size'],
            hidden_size=self.config['n_embd'],
            num_hidden_layers=self.config['n_layer'],
            num_attention_heads=self.config['n_head'],
            pad_token_id=self.config['pad_token_id'],
            pruning=self.pruning_config  # Add pruning config
        )
        
        self.model = IndicSLM(self.model_config)
        
        # Initialize pruner if pruning is enabled
        self.pruner = None
        if pruning_config and pruning_config.get('enabled', False):
            pruning_conf = PruningConfig(
                enabled=pruning_config.get('enabled', False),
                method=pruning_config.get('method', 'magnitude'),
                sparsity=pruning_config.get('sparsity', 0.3),
                schedule=pruning_config.get('schedule', 'gradual'),
                start_epoch=pruning_config.get('start_epoch', 0),
                end_epoch=pruning_config.get('end_epoch', 10),
                frequency=pruning_config.get('frequency', 1)
            )
            self.pruner = ModelPruner(self.model, pruning_conf)
        
        # Initialize tokenizer
        self.tokenizer = IndicTokenizer(config_path)
          # Training settings
        self.learning_rate = self.config['training']['learning_rate']
        self.warmup_steps = 100  # Reduced warmup steps
        self.max_steps = 1000    # Reduced max steps
    
    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch.get("labels")
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        
        # Log memory usage
        if batch_idx % 10 == 0:  # More frequent logging
            memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
            self.log('memory_usage_gb', memory_gb, prog_bar=True)
            
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def on_epoch_end(self):
        """Apply pruning at the end of each epoch if enabled"""
        if self.pruner:
            current_epoch = self.current_epoch
            if self.pruner.should_prune_on_epoch(current_epoch):
                # Apply pruning
                sparsity = self.pruner.apply_pruning(current_epoch)
                # Log pruning statistics
                self.log('sparsity', sparsity, prog_bar=True)
                # Get detailed pruning stats
                stats = self.pruner.get_pruning_statistics()
                for layer_name, layer_stats in stats.get('layer_sparsity', {}).items():
                    self.log(f'sparsity/{layer_name}', layer_stats['sparsity'])
                
                logger.info(f"Epoch {current_epoch}: Applied pruning with sparsity {sparsity:.4f}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

def train(
    config_path: str = str(CONFIG_PATH),
    output_dir: str = str(OUTPUT_DIR),
    max_epochs: int = 3,
    batch_size: int = 2,  # Ultra-small batch size
    accumulate_grad_batches: int = 16,  # Increased gradient accumulation
    num_workers: int = 0,  # No extra workers
    max_memory_gb: float = 8.0,  # Lower memory threshold
    dry_run: bool = False,  # Dry run option for testing
    enable_pruning: bool = False,  # Enable model pruning
    sparsity: float = 0.3,  # Target sparsity for pruning
    pruning_method: str = "magnitude",  # Pruning method
    pruning_schedule: str = "gradual",  # Pruning schedule
    pruning_start_epoch: int = 1,  # Starting epoch for pruning
    pruning_end_epoch: int = None  # Ending epoch for pruning
) -> None:
    """Ultra memory-efficient training function"""
    
    logger.info(f"Using config from: {config_path}")
    logger.info(f"Saving output to: {output_dir}")
    logger.info(f"Gradient accumulation steps: {accumulate_grad_batches}")
    
    if dry_run:
        logger.info("Running in dry-run mode (limited training for testing only)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Initialize WandB (skip in dry-run mode)
    if not dry_run:
        try:
            wandb.init(
                project="indic-slm",
                name="telugu-hindi-summarizer-efficient",
                config=config
            )
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}")
            logger.warning("Training will continue without WandB logging")
    
    logger.info("Loading tokenizer...")
    tokenizer = IndicTokenizer(config_path)
    
    # Create simple dataset that doesn't use multiprocessing
    logger.info("Creating simple dataset...")
    train_dataset = SimpleDataset(tokenizer, lang="te", max_length=128)
    
    # Create dataloader without multiprocessing
    logger.info("Creating data loader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable pin memory to reduce memory usage
    )
      # Initialize model
    logger.info("Initializing model...")
    
    # Setup pruning configuration if enabled
    pruning_config = None
    if enable_pruning:
        logger.info(f"Enabling pruning with {sparsity:.2f} sparsity using {pruning_method} method")
        pruning_config = {
            'enabled': enable_pruning,
            'method': pruning_method,
            'sparsity': sparsity,
            'schedule': pruning_schedule,
            'start_epoch': pruning_start_epoch,
            'end_epoch': pruning_end_epoch if pruning_end_epoch is not None else max_epochs - 1,
            'frequency': 1  # Apply pruning every epoch
        }
    
    model = IndicSLMTrainer(config_path, pruning_config=pruning_config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="indic-slm-{epoch:02d}-{train_loss:.2f}",
            save_top_k=1,  # Only keep the best model
            monitor="train_loss",
            mode="min",
            every_n_train_steps=100 if not dry_run else 5  # Save more frequently in dry run mode
        ),
        EarlyStopping(
            monitor="train_loss",
            patience=3,
            mode="min"
        ),
        MemoryMonitorCallback()
    ]
    
    # Initialize trainer with ultra memory-efficient settings
    trainer = pl.Trainer(
        max_epochs=1 if dry_run else max_epochs,  # Limit to 1 epoch for dry run
        accelerator="cpu",  # Force CPU training
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=pl.loggers.CSVLogger(output_dir),  # Use CSV logger instead of WandB
        precision="32", # Use 32-bit precision on CPU
        enable_checkpointing=True,
        log_every_n_steps=5 if dry_run else 10,  # More frequent logging in dry run mode
        limit_train_batches=5 if dry_run else 20,  # Limit training batches even more in dry run mode
        enable_model_summary=False,  # Disable model summary to save memory
        enable_progress_bar=True
    )
    
    logger.info(f"Starting training with gradient accumulation steps: {accumulate_grad_batches}")
    logger.info(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    
    try:
        trainer.fit(model, train_loader)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.error("Attempting to save partial model...")
    
    # Only save final model if not in dry-run mode
    if not dry_run:
        try:
            logger.info("Saving final model...")
            os.makedirs(os.path.join(output_dir, "final_model"), exist_ok=True)
            model.model.save_pretrained(os.path.join(output_dir, "final_model"))
            logger.info("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    else:
        logger.info("Skipping final model save in dry-run mode")
    
    try:
        if not dry_run and "wandb" in sys.modules:
            wandb.finish()
    except:
        pass

def generate_summary(text: str, lang: str, model_path: str, max_length: int = 128) -> str:
    """Generate summary for given text"""
    try:
        # Load model and tokenizer
        model = IndicSLMTrainer.load_from_checkpoint(model_path)
        model.eval()
        
        # Prepare input
        text = f"<{lang}> {text}"
        inputs = model.tokenizer.encode(text, return_tensors="pt")
        
        input_ids = inputs["input_ids"][:, :max_length]
        attention_mask = inputs["attention_mask"][:, :max_length]
        
        # Generate summary
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_beams=2,  # Reduce beam size
                length_penalty=1.0,
                early_stopping=True
            )
            
        summary = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"

if __name__ == "__main__":
    # Configure logging to file
    file_handler = logging.FileHandler(BASE_DIR / "train_output.log")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info("Starting training script...")
      # Define command-line arguments
    parser = argparse.ArgumentParser(description="Train the SLM model")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to configuration file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, 
                        help="Number of steps to accumulate gradients before optimizer step")
    parser.add_argument("--workers", type=int, default=0, 
                        help="Number of worker processes for data loading")
    parser.add_argument("--memory_threshold", type=float, default=8.0, 
                        help="Memory threshold in GB for auto-collection")
    parser.add_argument("--dry_run", action="store_true", 
                        help="Perform a dry run (limited training to test functionality)")
    parser.add_argument("--enable_pruning", action="store_true",
                        help="Enable model pruning during training")
    parser.add_argument("--sparsity", type=float, default=0.3,
                        help="Target sparsity ratio for pruning (0.0 to 1.0)")
    parser.add_argument("--pruning_method", type=str, default="magnitude", choices=["magnitude", "random", "structured"],
                        help="Pruning method to use")
    parser.add_argument("--pruning_schedule", type=str, default="gradual", choices=["one-shot", "gradual"],
                        help="Pruning schedule (one-shot or gradual)")
    parser.add_argument("--pruning_start_epoch", type=int, default=1,
                        help="Starting epoch for gradual pruning")
    parser.add_argument("--pruning_end_epoch", type=int, default=None,
                        help="Ending epoch for gradual pruning")
    
    args = parser.parse_args()
    
    # Start training with provided arguments
    train(
        config_path=args.config,
        output_dir=args.output,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        num_workers=args.workers,
        max_memory_gb=args.memory_threshold,
        dry_run=args.dry_run,
        enable_pruning=args.enable_pruning,
        sparsity=args.sparsity,
        pruning_method=args.pruning_method,
        pruning_schedule=args.pruning_schedule,
        pruning_start_epoch=args.pruning_start_epoch
    )