import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hf_dataset_download.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Set up base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset configurations
DATASET_CONFIGS = {
    "xlsum": {
        "name": "XL-Sum - Cross-lingual summarization dataset from BBC",
        "hf_path": "csebuetnlp/xlsum",
        "languages": {
            "te": {"name": "telugu", "description": "Telugu BBC News articles with summaries"},
            "hi": {"name": "hindi", "description": "Hindi BBC News articles with summaries"}
        },
        "text_field": "text",
        "summary_field": "summary"
    },
    "cnn_dailymail": {
        "name": "CNN/Daily Mail summarization dataset",
        "hf_path": "cnn_dailymail",
        "hf_version": "3.0.0", 
        "languages": {
            "en": {"name": "english", "description": "English news articles with summaries"}
        },
        "text_field": "article",
        "summary_field": "highlights"
    },
    "wiki_lingua": {
        "name": "WikiLingua dataset",
        "hf_path": "GEM/wiki_lingua",
        "languages": {
            "hi": {"name": "hi", "description": "Hindi WikiHow articles with summaries"},
        },
        "text_field": "source",
        "summary_field": "target"
    },
    "indic_nlp": {
        "name": "IndicNLP suite dataset (processed from local data)",
        "local_path": True,
        "languages": {
            "hi": {"name": "hindi", "description": "Hindi text for NLG tasks"},
            "te": {"name": "telugu", "description": "Telugu text for NLG tasks"}
        }
    }
}

def download_hf_dataset(dataset_name: str, lang: str, cache_dir: Path = None) -> Union[Dataset, DatasetDict]:
    """
    Download a dataset from Hugging Face
    
    Args:
        dataset_name: Name of the dataset
        lang: Language code
        cache_dir: Directory to cache the dataset
        
    Returns:
        The loaded dataset from Hugging Face
    """
    if dataset_name not in DATASET_CONFIGS:
        logger.error(f"Dataset {dataset_name} not found in configurations")
        return None
        
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    if lang not in dataset_config["languages"]:
        logger.error(f"Language {lang} not available for dataset {dataset_name}")
        return None
        
    lang_config = dataset_config["languages"][lang]
    
    logger.info(f"Downloading {dataset_name} for {lang}")
    logger.info(f"Description: {lang_config['description']}")
    
    try:
        # Handle special cases for different datasets
        if dataset_name == "xlsum":
            # XL-Sum requires the language as a config
            dataset = load_dataset(
                dataset_config["hf_path"], 
                lang_config["name"],
                cache_dir=cache_dir
            )
            logger.info(f"Successfully loaded {dataset_name} ({lang}) from Hugging Face")
            return dataset
            
        elif dataset_name == "cnn_dailymail":
            # CNN/DailyMail has version specification
            dataset = load_dataset(
                dataset_config["hf_path"], 
                dataset_config["hf_version"],
                cache_dir=cache_dir
            )
            logger.info(f"Successfully loaded {dataset_name} from Hugging Face")
            return dataset
            
        elif dataset_name == "wiki_lingua":
            # WikiLingua needs language-specific handling
            dataset = load_dataset(
                dataset_config["hf_path"],
                cache_dir=cache_dir
            )
            # Extract specific language if needed
            if hasattr(dataset, 'filter'):
                dataset = dataset.filter(lambda x: x['target_language'] == lang_config["name"])
            logger.info(f"Successfully loaded {dataset_name} ({lang}) from Hugging Face")
            return dataset
            
        # Generic case
        else:
            dataset = load_dataset(
                dataset_config["hf_path"],
                cache_dir=cache_dir
            )
            logger.info(f"Successfully loaded {dataset_name} from Hugging Face")
            return dataset
            
    except Exception as e:
        logger.error(f"Error downloading {dataset_name} for {lang} from Hugging Face: {e}")
        return None

def process_and_save_dataset(dataset: Union[Dataset, DatasetDict], dataset_name: str, lang: str, output_dir: Path) -> None:
    """
    Process a dataset and save it in a format suitable for training
    
    Args:
        dataset: The loaded dataset
        dataset_name: Name of the dataset
        lang: Language code
        output_dir: Directory to save the processed files
    """
    if dataset is None:
        logger.error(f"Cannot process empty dataset for {dataset_name} ({lang})")
        return
        
    # Create output directories for train/val/test
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    logger.info(f"Processing {dataset_name} for {lang}")
    
    # Get the field names for text and summary from config
    dataset_config = DATASET_CONFIGS[dataset_name]
    text_field = dataset_config.get("text_field", "text")
    summary_field = dataset_config.get("summary_field", "summary")
    
    # Check if dataset is a DatasetDict with splits or a single Dataset
    if isinstance(dataset, DatasetDict):
        splits = dataset.keys()
        for split_name in splits:
            # Map split names to our standard formats
            if split_name in ["train", "training"]:
                output_split_dir = train_dir
            elif split_name in ["validation", "val", "dev"]:
                output_split_dir = val_dir
            elif split_name in ["test", "testing"]:
                output_split_dir = test_dir
            else:
                logger.warning(f"Unknown split name: {split_name}, skipping")
                continue
                
            split_dataset = dataset[split_name]
            save_split(split_dataset, split_name, output_split_dir, text_field, summary_field, lang, dataset_name)
    else:
        # Single dataset without splits - allocate to train
        logger.warning(f"Dataset {dataset_name} has no splits, saving all to train")
        save_split(dataset, "train", train_dir, text_field, summary_field, lang, dataset_name)
        
    logger.info(f"Finished processing {dataset_name} for {lang}")

def save_split(dataset: Dataset, split_name: str, output_dir: Path, text_field: str, summary_field: str,
               lang: str, dataset_name: str) -> None:
    """
    Save a dataset split as text and summary files
    
    Args:
        dataset: The dataset split
        split_name: Name of the split
        output_dir: Directory to save the files
        text_field: Field name for the source text
        summary_field: Field name for the summary
        lang: Language code
        dataset_name: Name of the dataset
    """
    logger.info(f"Saving {split_name} split with {len(dataset)} examples to {output_dir}")
    
    samples_processed = 0
    skipped = 0
    
    # Process each example in the dataset
    for i, example in enumerate(tqdm(dataset, desc=f"Processing {split_name} split")):
        try:
            doc_id = f"{dataset_name}_{lang}_{split_name}_{i}"
            
            # Get text and summary with fallbacks
            text = example.get(text_field, "")
            summary = example.get(summary_field, "")
            
            # Skip if either text or summary is missing
            if not text or not summary:
                skipped += 1
                continue
                
            # Save text and summary as separate files
            text_file = output_dir / f"{doc_id}.txt"
            summary_file = output_dir / f"{doc_id}.summary"
            
            with open(text_file, 'w', encoding='utf-8') as tf:
                tf.write(text)
                
            with open(summary_file, 'w', encoding='utf-8') as sf:
                sf.write(summary)
                
            samples_processed += 1
                
        except Exception as e:
            logger.error(f"Error processing example {i} in {split_name} split: {e}")
            
    logger.info(f"Processed {samples_processed} samples for {split_name} split (skipped {skipped})")

def process_local_data(dataset_name: str, lang: str, output_dir: Path) -> None:
    """
    Process data from local files
    
    Args:
        dataset_name: Name of the dataset
        lang: Language code
        output_dir: Directory to save the processed files
    """
    logger.info(f"Processing local data for {dataset_name} ({lang})")
    
    # Look for data in expected locations
    local_data_path = RAW_DATA_DIR / lang
    
    if not os.path.exists(local_data_path):
        logger.error(f"Local data directory not found for {lang}: {local_data_path}")
        return
        
    # Create output directories
    train_dir = output_dir / "train"
    os.makedirs(train_dir, exist_ok=True)
    
    # Process files found in the local directory
    processed = 0
    
    # Simple heuristic: look for text files with potential summaries
    text_files = list(local_data_path.glob("*.txt"))
    
    for text_file in tqdm(text_files, desc=f"Processing local {lang} files"):
        try:
            # Look for corresponding summary file
            summary_file = text_file.with_suffix('.summary')
            if not os.path.exists(summary_file):
                # Try alternatives
                summary_candidates = [
                    text_file.parent / f"{text_file.stem}.summary",
                    text_file.parent / f"{text_file.stem}_summary.txt",
                    text_file.parent / f"{text_file.stem}.sum"
                ]
                
                for candidate in summary_candidates:
                    if os.path.exists(candidate):
                        summary_file = candidate
                        break
                else:
                    # No summary file found, skip
                    continue
                    
            # Read text and summary
            with open(text_file, 'r', encoding='utf-8') as tf:
                text = tf.read().strip()
                
            with open(summary_file, 'r', encoding='utf-8') as sf:
                summary = sf.read().strip()
                
            # Skip if either is empty
            if not text or not summary:
                continue
                
            # Save to output directory
            output_text_file = train_dir / f"{dataset_name}_{lang}_local_{processed}.txt"
            output_summary_file = train_dir / f"{dataset_name}_{lang}_local_{processed}.summary"
            
            with open(output_text_file, 'w', encoding='utf-8') as otf:
                otf.write(text)
                
            with open(output_summary_file, 'w', encoding='utf-8') as osf:
                osf.write(summary)
                
            processed += 1
                
        except Exception as e:
            logger.error(f"Error processing local file {text_file}: {e}")
            
    logger.info(f"Processed {processed} local samples for {dataset_name} ({lang})")

def create_combined_dataset(output_file: Path, langs: List[str]) -> None:
    """
    Create a combined dataset file from processed files for training
    
    Args:
        output_file: Path to save the combined dataset
        langs: List of language codes to include
    """
    logger.info(f"Creating combined dataset for languages: {', '.join(langs)}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # Process each dataset
        for dataset_name in DATASET_CONFIGS:
            for lang in langs:
                if lang not in DATASET_CONFIGS[dataset_name]["languages"]:
                    continue
                
                dataset_dir = PROCESSED_DATA_DIR / dataset_name / lang / "train"
                
                if not os.path.exists(dataset_dir):
                    logger.warning(f"Directory not found: {dataset_dir}")
                    continue
                
                # Find all text files
                text_files = list(dataset_dir.glob("*.txt"))
                logger.info(f"Found {len(text_files)} samples in {dataset_dir}")
                
                # Process each text file
                for text_file in tqdm(text_files, desc=f"Processing {dataset_name} {lang}"):
                    summary_file = text_file.with_suffix('.summary')
                    
                    if not os.path.exists(summary_file):
                        continue
                        
                    try:
                        with open(text_file, 'r', encoding='utf-8') as tf:
                            text = tf.read().strip()
                            
                        with open(summary_file, 'r', encoding='utf-8') as sf:
                            summary = sf.read().strip()
                            
                        # Skip empty samples
                        if not text or not summary:
                            continue
                            
                        # Add language tag
                        entry = {
                            "text": f"<{lang}> {text}",
                            "summary": summary,
                            "lang": lang,
                            "source": dataset_name
                        }
                        
                        # Write to output file
                        out_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        
                    except Exception as e:
                        logger.error(f"Error processing {text_file}: {e}")
                        
    logger.info(f"Combined dataset created at {output_file}")

def main():
    """Main function to download and process datasets"""
    parser = argparse.ArgumentParser(description="Download and process datasets for multilingual summarization using Hugging Face")
    
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs="+", 
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Datasets to download (default: all)"
    )
    
    parser.add_argument(
        "--langs", 
        type=str, 
        nargs="+", 
        choices=["te", "hi", "en", "all"],
        default=["all"],
        help="Languages to download (default: all)"
    )
    
    parser.add_argument(
        "--download-only", 
        action="store_true",
        help="Only download datasets without processing"
    )
    
    parser.add_argument(
        "--process-only", 
        action="store_true",
        help="Only process already downloaded datasets"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        default=str(DATA_DIR / "combined_dataset.jsonl"),
        help="Path to save the combined dataset"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str,
        default=str(CACHE_DIR),
        help="Directory to cache the downloaded datasets"
    )
    
    args = parser.parse_args()
    
    # Handle "all" option for datasets
    if "all" in args.datasets:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets
        
    # Handle "all" option for languages
    if "all" in args.langs:
        langs = ["te", "hi"]
    else:
        langs = args.langs
        
    logger.info(f"Selected datasets: {', '.join(datasets)}")
    logger.info(f"Selected languages: {', '.join(langs)}")
    
    # Download and process datasets
    downloaded_datasets = {}
    
    # Download datasets using Hugging Face
    if not args.process_only:
        for dataset_name in datasets:
            dataset_config = DATASET_CONFIGS[dataset_name]
            
            # Skip local data sources for download
            if dataset_config.get("local_path", False):
                logger.info(f"Skipping download for local dataset: {dataset_name}")
                continue
                
            for lang in langs:
                if lang not in dataset_config["languages"]:
                    continue
                    
                try:
                    # Download from Hugging Face
                    dataset = download_hf_dataset(
                        dataset_name, 
                        lang, 
                        cache_dir=Path(args.cache_dir)
                    )
                    
                    if dataset:
                        # Store for processing
                        if dataset_name not in downloaded_datasets:
                            downloaded_datasets[dataset_name] = {}
                        downloaded_datasets[dataset_name][lang] = dataset
                        
                except Exception as e:
                    logger.error(f"Error downloading {dataset_name} for {lang}: {e}")
    
    # Process datasets
    if not args.download_only:
        for dataset_name in datasets:
            dataset_config = DATASET_CONFIGS[dataset_name]
            
            for lang in langs:
                if lang not in dataset_config["languages"]:
                    continue
                    
                try:
                    output_dir = PROCESSED_DATA_DIR / dataset_name / lang
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Check if this is a local dataset
                    if dataset_config.get("local_path", False):
                        process_local_data(dataset_name, lang, output_dir)
                    elif dataset_name in downloaded_datasets and lang in downloaded_datasets[dataset_name]:
                        # Process downloaded dataset
                        process_and_save_dataset(
                            downloaded_datasets[dataset_name][lang],
                            dataset_name,
                            lang,
                            output_dir
                        )
                    else:
                        logger.warning(f"No downloaded data available for {dataset_name} ({lang})")
                        
                except Exception as e:
                    logger.error(f"Error processing {dataset_name} for {lang}: {e}")
        
        # Create combined dataset
        create_combined_dataset(Path(args.output), langs)
    
    logger.info("Download and processing complete!")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time}")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info(f"Script completed at {end_time}")
    logger.info(f"Total elapsed time: {elapsed_time}")