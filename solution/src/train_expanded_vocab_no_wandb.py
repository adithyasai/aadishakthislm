import os
import sys
import json
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
import gc
import psutil

# Set this to disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer
from src.data_processor import DataProcessor

# Setup paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "models" / "expanded_vocab_model"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedVocabDataset(Dataset):
    """Dataset class that uses real multilingual data for training the expanded vocab model"""
    
    def __init__(self, tokenizer, data_paths=None, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load sample data from files if provided, otherwise use default samples
        if data_paths and len(data_paths) > 0:
            for path in data_paths:
                self._load_data_from_file(path)
            logger.info(f"Loaded {len(self.samples)} samples from provided data paths")
        else:
            self._load_default_samples()
            logger.info(f"Loaded {len(self.samples)} default samples")
    
    def _load_data_from_file(self, file_path):
        """Load data from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Detect language or use metadata if available
                        # For simplicity, we're just using the line as is
                        self.samples.append(line)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
    
    def _load_default_samples(self):
        """Load default multilingual samples for training"""
        # Hindi samples
        hindi_samples = [
            "हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है।",
            "भारत में कई भाषाएँ बोली जाती हैं। हिंदी उत्तर भारत में सबसे अधिक बोली जाती है।",
            "भारतीय संविधान ने हिंदी को भारत की आधिकारिक भाषा के रूप में मान्यता दी है।"
        ]
        
        # Telugu samples
        telugu_samples = [
            "తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష.",
            "తెలుగు భాష భారతదేశంలోని ప్రాచీన భాషలలో ఒకటి మరియు సంస్కృత భాష తర్వాత అత్యంత సాహిత్యపరమైన భాష.",
            "తెలుగు భాషా దినోత్సవం ప్రతి సంవత్సరం ఆగస్టు 29న జరుపుకుంటారు."
        ]
        
        # Bengali samples
        bengali_samples = [
            "বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গ রাজ্যের প্রধান ভাষা।",
            "বাংলা ভাষা একটি ইন্দো-আর্য ভাষা যা ভারতীয় উপমহাদেশে প্রায় ২৩০ মিলিয়ন লোক কথ্য ভাষা হিসাবে ব্যবহার করে।",
            "রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যের একজন বিখ্যাত কবি এবং লেখক।"
        ]
        
        # Malayalam samples
        malayalam_samples = [
            "മലയാളം ഇന്ത്യയിലെ കേരള സംസ്ഥാനത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ്.",
            "മലയാളം ദ്രാവിഡ ഭാഷാ കുടുംബത്തിൽ ഉൾപ്പെടുന്നു, തമിഴ്, കന്നഡ, തെലുങ്ക് എന്നിവയുമായി ബന്ധപ്പെട്ടിരിക്കുന്നു.",
            "കേരളത്തിൽ ഏകദേശം 38 ദശലക്ഷം ആളുകൾ മലയാളം സംസാരിക്കുന്നു."
        ]
        
        # Kannada samples
        kannada_samples = [
            "ಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.",
            "ಕನ್ನಡ ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ.",
            "ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಸುಮಾರು 1500 ವರ್ಷಗಳ ಇತಿಹಾಸವನ್ನು ಹೊಂದಿದೆ."
        ]
        
        # Add samples with language tags
        for sample in hindi_samples:
            self.samples.append(f"<hi> {sample}")
        
        for sample in telugu_samples:
            self.samples.append(f"<te> {sample}")
        
        for sample in bengali_samples:
            self.samples.append(f"<bn> {sample}")
        
        for sample in malayalam_samples:
            self.samples.append(f"<ml> {sample}")
        
        for sample in kannada_samples:
            self.samples.append(f"<kn> {sample}")
        
        # Repeat samples to create a larger dataset
        self.samples = self.samples * 10  # Multiply by 10 to get more training data
    
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

class MemoryEfficientCallback(pl.Callback):
    def __init__(self, memory_threshold_gb=8.0):
        super().__init__()
        self.memory_threshold_gb = memory_threshold_gb
    
    def on_batch_end(self, trainer, pl_module):
        memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        if memory_gb > self.memory_threshold_gb:
            logger.warning(f"High memory usage detected: {memory_gb:.2f}GB")
            gc.collect()
            torch.cuda.empty_cache()

class ExpandedVocabSLMTrainer(pl.LightningModule):
    def __init__(self, config_path: str = str(CONFIG_PATH)):
        super().__init__()
        self.save_hyperparameters()
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Get configuration values with fallbacks
        hidden_size = self.config.get('n_embd', 384)
        num_layers = self.config.get('n_layer', 6)
        num_heads = self.config.get('n_head', 6)
        vocab_size = self.config.get('vocab_size', 50000)  # Use our expanded vocab size
        pad_token_id = self.config.get('pad_token_id', 0)
        
        logger.info(f"Initializing model with vocab_size={vocab_size}, hidden_size={hidden_size}")
        
        # Create model configuration
        self.model_config = IndicSLMConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            pad_token_id=pad_token_id
        )
        
        # Initialize model
        self.model = IndicSLM(self.model_config)
        
        # Initialize tokenizer
        self.tokenizer = IndicTokenizer(config_path)
        
        # Log the actual tokenizer vocabulary size
        logger.info(f"Tokenizer vocabulary size: {self.tokenizer.sp_model.get_piece_size()}")
        
        # Training settings
        training_config = self.config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 5e-5)
        self.warmup_steps = training_config.get('warmup_steps', 100)
        self.max_steps = training_config.get('max_steps', 1000)
    
    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels")
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        
        # Log memory usage occasionally
        if batch_idx % 10 == 0:
            memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
            self.log('memory_usage_gb', memory_gb, prog_bar=True)
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Linear learning rate scheduler with warmup
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

def train_expanded_vocab_model(
    config_path: str = str(CONFIG_PATH),
    output_dir: str = str(OUTPUT_DIR),
    data_paths: list = None,
    max_epochs: int = 5,
    batch_size: int = 4,
    accumulate_grad_batches: int = 8,
    num_workers: int = 2,
    memory_threshold_gb: float = 8.0,
    use_gpu: bool = False,
    use_wandb: bool = False
):
    """Train the SLM model with expanded vocabulary"""
    
    logger.info(f"Starting training with expanded vocabulary (50,000 tokens)")
    logger.info(f"Using config from: {config_path}")
    logger.info(f"Saving output to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging to file
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Initialize WandB for experiment tracking (only if requested)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="indic-slm-expanded-vocab",
                name=f"expanded-vocab-model-{config['vocab_size']}",
                config=config
            )
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}")
            logger.warning("Training will continue without WandB logging")
            use_wandb = False
    
    # Initialize tokenizer
    logger.info("Loading tokenizer with expanded vocabulary...")
    tokenizer = IndicTokenizer(config_path)
    
    # Create dataset
    logger.info("Creating dataset with multilingual samples...")
    dataset = ExpandedVocabDataset(tokenizer, data_paths=data_paths, max_length=256)
    
    # Split dataset into train and validation
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    else:
        train_dataset = dataset
        val_dataset = None
    
    logger.info(f"Dataset created with {len(train_dataset)} training samples")
    if val_dataset:
        logger.info(f"Validation set has {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_gpu
        )
    
    # Initialize model
    logger.info("Initializing model with expanded vocabulary...")
    model = ExpandedVocabSLMTrainer(config_path)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="indic-slm-expanded-vocab-{epoch:02d}-{train_loss:.4f}",
            save_top_k=2,
            monitor="train_loss",
            mode="min",
            every_n_train_steps=100
        ),
        EarlyStopping(
            monitor="train_loss" if val_loader is None else "val_loss",
            patience=5,
            mode="min"
        ),
        MemoryEfficientCallback(memory_threshold_gb=memory_threshold_gb)
    ]
    
    # Determine training device
    accelerator = "gpu" if use_gpu and torch.cuda.is_available() else "cpu"
    if accelerator == "gpu":
        logger.info("Using GPU for training")
    else:
        logger.info("Using CPU for training")
    
    # Choose appropriate logger
    if use_wandb:
        import wandb
        logger_to_use = pl.loggers.WandbLogger()
    else:
        logger_to_use = pl.loggers.CSVLogger(output_dir)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger_to_use,
        precision="16-mixed" if accelerator == "gpu" else "32",
        enable_checkpointing=True,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Start training
    logger.info("Starting training process...")
    try:
        if val_loader:
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.error("Attempting to save partial model...")
    
    # Save the final model
    try:
        logger.info("Saving final model...")
        final_model_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        model.model.save_pretrained(final_model_dir)
        logger.info(f"Model saved successfully to {final_model_dir}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    # Cleanup
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
    
    return os.path.join(output_dir, "final_model")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train SLM model with expanded vocabulary")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to config file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--data", type=str, nargs="+", help="Paths to data files")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    args = parser.parse_args()
    
    # Train the model
    train_expanded_vocab_model(
        config_path=args.config,
        output_dir=args.output,
        data_paths=args.data,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        use_gpu=args.gpu,
        use_wandb=args.wandb
    )