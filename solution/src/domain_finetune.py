#!/usr/bin/env python3
"""
Domain-specific fine-tuning for SLM using parameter-efficient LoRA adapters.

Requirements:
  pip install transformers datasets peft accelerate
"""
# Ensure accelerate.utils.memory.clear_device_cache exists for PEFT
try:
    import accelerate.utils.memory as _acc_mem
    if not hasattr(_acc_mem, "clear_device_cache"):
        setattr(_acc_mem, "clear_device_cache", lambda *args, **kwargs: None)
except ImportError:
    pass

import os
import argparse
from pathlib import Path
import logging
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(os.getenv("DEFAULT_MODEL_PATH", "models/final_model"))


def parse_args():
    parser = argparse.ArgumentParser(description="Domain-specific fine-tuning with LoRA adapters")
    parser.add_argument(
        "--domain", type=str, required=True,
        choices=["legal", "medical", "technical"],
        help="Domain for fine-tuning"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to JSONL dataset (fields: text, summary)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run data loading and preprocessing only"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Learning rate"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load JSON lines dataset manually
    logger.info(f"Loading dataset from {args.data_path}")
    raw_data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_data.append(json.loads(line))
    logger.info(f"Loaded {len(raw_data)} examples")

    # If dry run, exit before heavy imports
    if args.dry_run:
        logger.info("Dry run complete: data loaded.")
        return

    # Load tokenizer for preprocessing
    logger.info(f"Loading tokenizer and model from {DEFAULT_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_PATH))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(DEFAULT_MODEL_PATH))

    # Preprocessing and tokenization
    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["summary"]
        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            targets, max_length=128, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing dataset")
    # Apply preprocessing to raw_data
    tokenized = [preprocess_function(ex) for ex in raw_data]

    # Import training and PEFT modules after dry_run check to avoid version mismatches
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType

    # Apply LoRA
    logger.info("Applying LoRA adapter configuration")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)

    # Training arguments
    output_dir = Path(args.output_dir) / args.domain
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    logger.info("Starting fine-tuning")
    trainer.train()

    # Save adapter and model
    logger.info(f"Saving fine-tuned model to {output_dir}")
    model.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
