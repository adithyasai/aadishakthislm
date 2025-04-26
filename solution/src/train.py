import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import get_linear_schedule_with_warmup
import wandb
from typing import Dict, Any, Optional
import os
import json

from model import IndicSLM, IndicSLMConfig
from tokenizer import IndicTokenizer
from data_processor import DataProcessor

class IndicSLMTrainer(pl.LightningModule):
    def __init__(self, config_path: str = "../configs/model_config.json"):
        super().__init__()
        self.save_hyperparameters()
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.model_config = IndicSLMConfig(**self.config)
        self.model = IndicSLM(self.model_config)
        
        # Initialize tokenizer
        self.tokenizer = IndicTokenizer(config_path)
        
        # Training settings
        self.learning_rate = self.config['training']['learning_rate']
        self.warmup_steps = self.config['training']['warmup_steps']
        self.max_steps = self.config['training']['max_steps']
        
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(**batch)
        val_loss = outputs['loss']
        self.log('val_loss', val_loss, prog_bar=True)
        
        # Log sample predictions
        if batch_idx == 0:
            input_text = self.tokenizer.decode(batch['input_ids'][0].tolist())
            pred_ids = torch.argmax(outputs['logits'][0], dim=-1)
            pred_text = self.tokenizer.decode(pred_ids.tolist())
            self.logger.experiment.log({
                "examples": wandb.Table(
                    data=[[input_text, pred_text]],
                    columns=["input", "prediction"]
                )
            })
    
    def configure_optimizers(self):
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler
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
    config_path: str = "../configs/model_config.json",
    train_data_dir: str = "../data/processed",
    output_dir: str = "../models",
    max_epochs: int = 10
) -> None:
    """Main training function"""
    
    # Initialize wandb logging
    wandb.init(project="indic-slm", name="telugu-hindi-summarizer")
    
    # Load data
    data_processor = DataProcessor(config_path)
    
    # Process Telugu data
    te_dataset = data_processor.prepare_dataset(
        os.path.join(train_data_dir, "telugu_train.csv"),
        lang="te"
    )
    
    # Process Hindi data
    hi_dataset = data_processor.prepare_dataset(
        os.path.join(train_data_dir, "hindi_train.csv"),
        lang="hi"
    )
    
    # Split datasets
    te_splits = data_processor.create_train_val_test_split(te_dataset)
    hi_splits = data_processor.create_train_val_test_split(hi_dataset)
    
    # Initialize model trainer
    trainer = IndicSLMTrainer(config_path)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="indic-slm-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
    ]
    
    # Initialize PyTorch Lightning trainer
    pl_trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=pl.loggers.WandbLogger(project="indic-slm")
    )
    
    # Start training
    pl_trainer.fit(
        trainer,
        train_dataloaders=DataLoader(
            te_splits["train"].concatenate(hi_splits["train"]),
            batch_size=trainer.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4
        ),
        val_dataloaders=DataLoader(
            te_splits["validation"].concatenate(hi_splits["validation"]),
            batch_size=trainer.config["training"]["batch_size"],
            num_workers=4
        )
    )
    
    # Save final model and tokenizer
    trainer.model.save_pretrained(os.path.join(output_dir, "final_model"))
    trainer.tokenizer.save_tokenizer(os.path.join(output_dir, "tokenizer"))
    
    # End wandb run
    wandb.finish()

if __name__ == "__main__":
    train()