import os
import sys
import logging
import argparse
import shutil
import json
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import requests
from tqdm import tqdm
import pandas as pd
import urllib.request
import gzip
import hashlib
import re
import multiprocessing
from multiprocessing import Pool
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_download.log"),
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
    "indiccorp": {
        "name": "IndicCorp - Large monolingual corpus for Indian languages",
        "languages": {
            "te": {
                "url": "https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/indic-corp-frozen/te.tar.xz",
                "file_type": "tar.xz",
                "size_gb": 1.2,
                "description": "Telugu monolingual corpus with 1.2B tokens"
            },
            "hi": {
                "url": "https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/indic-corp-frozen/hi.tar.xz",
                "file_type": "tar.xz",
                "size_gb": 8.9,
                "description": "Hindi monolingual corpus with 8.9B tokens"
            }
        }
    },
    "xlsum": {
        "name": "XL-Sum - Cross-lingual summarization dataset",
        "languages": {
            "te": {
                "url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/telugu_train.jsonl.gz",
                "train_url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/telugu_train.jsonl.gz",
                "val_url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/telugu_val.jsonl.gz",
                "test_url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/telugu_test.jsonl.gz",
                "file_type": "jsonl.gz",
                "size_gb": 0.02,
                "description": "BBC articles and their summaries in Telugu"
            },
            "hi": {
                "url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/hindi_train.jsonl.gz",
                "train_url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/hindi_train.jsonl.gz",
                "val_url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/hindi_val.jsonl.gz",
                "test_url": "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/hindi_test.jsonl.gz",
                "file_type": "jsonl.gz",
                "size_gb": 0.1,
                "description": "BBC articles and their summaries in Hindi"
            }
        }
    },
    "wiki-lingua": {
        "name": "WikiLingua - Cross-lingual summarization dataset",
        "languages": {
            "hi": {
                "url": "https://github.com/esdurmus/Wikilingua/raw/master/data/hindi.zip",
                "file_type": "zip",
                "size_gb": 0.05,
                "description": "WikiHow articles and their summaries in Hindi"
            }
        }
    },
    "samanantar": {
        "name": "Samanantar - Parallel corpora for Indian languages",
        "languages": {
            "te": {
                "url": "http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/indic_wat_2021.zip",
                "file_type": "zip",
                "size_gb": 0.5,
                "description": "Parallel corpus for Telugu-English"
            },
            "hi": {
                "url": "http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/indic_wat_2021.zip",
                "file_type": "zip",
                "size_gb": 0.5,
                "description": "Parallel corpus for Hindi-English"
            }
        }
    },
    "indicnlg": {
        "name": "IndicNLG Suite - Dataset for natural language generation",
        "languages": {
            "te": {
                "url": "https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indicnlg/v1/indicnlg-te.zip",
                "file_type": "zip",
                "size_gb": 0.01,
                "description": "Telugu data for NLG tasks including summarization"
            },
            "hi": {
                "url": "https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indicnlg/v1/indicnlg-hi.zip",
                "file_type": "zip",
                "size_gb": 0.02,
                "description": "Hindi data for NLG tasks including summarization"
            }
        }
    },
    "news-summary": {
        "name": "Indian Language News Corpus",
        "languages": {
            "te": {
                "url": "https://iittp.ac.in/~ai/resources/indic-news-corpus/te_news.zip",
                "file_type": "zip",
                "size_gb": 0.3,
                "description": "Telugu news articles with summaries"
            },
            "hi": {
                "url": "https://iittp.ac.in/~ai/resources/indic-news-corpus/hi_news.zip",
                "file_type": "zip",
                "size_gb": 0.5,
                "description": "Hindi news articles with summaries"
            }
        }
    }
}

def download_file(url: str, output_path: Path, desc: str = None) -> Path:
    """
    Download a file with progress bar
    
    Args:
        url: URL to download
        output_path: Path to save the file
        desc: Description for the progress bar
        
    Returns:
        Path to the downloaded file
    """
    if not desc:
        desc = f"Downloading {os.path.basename(url)}"
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If file already exists, verify its integrity or skip
    if os.path.exists(output_path):
        logger.info(f"File already exists at {output_path}, skipping download")
        return output_path
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(output_path, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
                
        if total_size != 0 and progress_bar.n != total_size:
            logger.warning("WARNING: Downloaded file size does not match expected size!")
            
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file from {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
        
def extract_file(file_path: Path, output_dir: Path) -> Path:
    """
    Extract a compressed file
    
    Args:
        file_path: Path to the compressed file
        output_dir: Directory to extract the contents to
        
    Returns:
        Path to the extracted directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_path_str = str(file_path)
    
    try:
        if file_path_str.endswith('.zip'):
            logger.info(f"Extracting zip file: {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                
        elif file_path_str.endswith('.tar.gz') or file_path_str.endswith('.tgz'):
            logger.info(f"Extracting tar.gz file: {file_path}")
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_dir)
                
        elif file_path_str.endswith('.tar.xz'):
            logger.info(f"Extracting tar.xz file: {file_path}")
            with tarfile.open(file_path, 'r:xz') as tar_ref:
                tar_ref.extractall(output_dir)
                
        elif file_path_str.endswith('.tar'):
            logger.info(f"Extracting tar file: {file_path}")
            with tarfile.open(file_path, 'r:') as tar_ref:
                tar_ref.extractall(output_dir)
                
        elif file_path_str.endswith('.gz') and not file_path_str.endswith('.tar.gz'):
            logger.info(f"Extracting gzip file: {file_path}")
            output_file = os.path.join(output_dir, os.path.basename(file_path_str[:-3]))
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            logger.warning(f"Unsupported archive format: {file_path}")
            return file_path
            
        return output_dir
        
    except (zipfile.BadZipFile, tarfile.ReadError, gzip.BadGzipFile) as e:
        logger.error(f"Error extracting file {file_path}: {e}")
        raise

def process_xlsum_dataset(lang: str, output_dir: Path) -> None:
    """
    Process the XL-Sum dataset
    
    Args:
        lang: Language code (hi, te)
        output_dir: Directory to save the processed files
    """
    logger.info(f"Processing XL-Sum dataset for {lang}")
    
    train_file = RAW_DATA_DIR / f"xlsum/{lang}_train.jsonl"
    val_file = RAW_DATA_DIR / f"xlsum/{lang}_val.jsonl"
    test_file = RAW_DATA_DIR / f"xlsum/{lang}_test.jsonl"
    
    if not os.path.exists(train_file) or not os.path.exists(val_file) or not os.path.exists(test_file):
        logger.error(f"XL-Sum files not found for {lang}")
        return
    
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val" 
    test_dir = output_dir / "test"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each split
    for split_name, file_path in [("train", train_file), ("val", val_file), ("test", test_file)]:
        split_dir = output_dir / split_name
        samples_processed = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {split_name} split"):
                try:
                    data = json.loads(line.strip())
                    doc_id = data.get('id', f"{lang}_{split_name}_{samples_processed}")
                    
                    # Save text and summary as separate files
                    text_file = split_dir / f"{doc_id}.txt"
                    summary_file = split_dir / f"{doc_id}.summary"
                    
                    with open(text_file, 'w', encoding='utf-8') as tf:
                        tf.write(data.get('text', ''))
                        
                    with open(summary_file, 'w', encoding='utf-8') as sf:
                        tf.write(data.get('summary', ''))
                        
                    samples_processed += 1
                    
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON line in {file_path}")
                except Exception as e:
                    logger.error(f"Error processing line in {file_path}: {e}")
        
        logger.info(f"Processed {samples_processed} samples for {split_name} split")

def process_indicnlg_dataset(lang: str, output_dir: Path) -> None:
    """
    Process the IndicNLG dataset
    
    Args:
        lang: Language code (hi, te)
        output_dir: Directory to save the processed files
    """
    logger.info(f"Processing IndicNLG dataset for {lang}")
    
    input_dir = RAW_DATA_DIR / f"indicnlg/indicnlg-{lang}"
    
    if not os.path.exists(input_dir):
        logger.error(f"IndicNLG directory not found for {lang}: {input_dir}")
        return
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Check for summarization data
    summ_dir = input_dir / "summarization"
    if not os.path.exists(summ_dir):
        logger.warning(f"Summarization directory not found in IndicNLG for {lang}")
        return
    
    # Process summarization data
    for split_name in ["train", "valid", "test"]:
        out_split = "val" if split_name == "valid" else split_name
        split_dir = output_dir / out_split
        
        input_file = summ_dir / f"{split_name}.jsonl"
        if not os.path.exists(input_file):
            logger.warning(f"File not found: {input_file}")
            continue
            
        samples_processed = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {split_name} split"):
                try:
                    data = json.loads(line.strip())
                    doc_id = data.get('id', f"indicnlg_{lang}_{split_name}_{samples_processed}")
                    
                    # Save text and summary as separate files
                    text_file = split_dir / f"{doc_id}.txt"
                    summary_file = split_dir / f"{doc_id}.summary"
                    
                    with open(text_file, 'w', encoding='utf-8') as tf:
                        tf.write(data.get('article', ''))
                        
                    with open(summary_file, 'w', encoding='utf-8') as sf:
                        tf.write(data.get('summary', ''))
                        
                    samples_processed += 1
                    
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON line in {input_file}")
                except Exception as e:
                    logger.error(f"Error processing line in {input_file}: {e}")
                    
        logger.info(f"Processed {samples_processed} samples for {split_name} split")

def download_dataset(dataset_name: str, lang: str, download_dir: Path) -> None:
    """
    Download a dataset for a specific language
    
    Args:
        dataset_name: Name of the dataset
        lang: Language code
        download_dir: Directory to save the downloaded files
    """
    if dataset_name not in DATASET_CONFIGS:
        logger.error(f"Dataset {dataset_name} not found in configurations")
        return
        
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    if lang not in dataset_config["languages"]:
        logger.error(f"Language {lang} not available for dataset {dataset_name}")
        return
        
    lang_config = dataset_config["languages"][lang]
    
    # Create dataset directory
    dataset_dir = download_dir / dataset_name
    os.makedirs(dataset_dir, exist_ok=True)
    
    logger.info(f"Downloading {dataset_name} for {lang}")
    logger.info(f"Description: {lang_config['description']}")
    logger.info(f"Size: {lang_config['size_gb']} GB")
    
    # Download based on dataset type
    if dataset_name == "xlsum":
        # XL-Sum has train, val, test splits
        for split, url_key in [
            ("train", "train_url"),
            ("val", "val_url"),
            ("test", "test_url")
        ]:
            if url_key in lang_config:
                url = lang_config[url_key]
                file_name = f"{lang}_{split}.jsonl.gz"
                output_path = dataset_dir / file_name
                
                # Download the file
                download_file(url, output_path, f"Downloading {dataset_name} {lang} {split}")
                
                # Extract if needed
                if output_path.suffix == '.gz':
                    extract_file(output_path, dataset_dir)
    else:
        # Generic download for other datasets
        url = lang_config["url"]
        file_name = os.path.basename(url)
        output_path = dataset_dir / file_name
        
        # Download the file
        download_file(url, output_path, f"Downloading {dataset_name} {lang}")
        
        # Extract if it's a compressed file
        file_type = lang_config["file_type"]
        if file_type in ["zip", "tar.gz", "tar.xz", "tgz", "gz"]:
            extract_dir = dataset_dir / f"{lang}"
            extract_file(output_path, extract_dir)
            
    logger.info(f"Finished downloading {dataset_name} for {lang}")

def process_dataset(dataset_name: str, lang: str) -> None:
    """
    Process a downloaded dataset for a specific language
    
    Args:
        dataset_name: Name of the dataset
        lang: Language code
    """
    logger.info(f"Processing {dataset_name} for {lang}")
    
    # Set up input and output directories
    input_dir = RAW_DATA_DIR / dataset_name
    output_dir = PROCESSED_DATA_DIR / dataset_name / lang
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process based on dataset type
    if dataset_name == "xlsum":
        process_xlsum_dataset(lang, output_dir)
    elif dataset_name == "indicnlg":
        process_indicnlg_dataset(lang, output_dir)
    # Add processing for other datasets as needed
    else:
        logger.info(f"No specific processing implemented for {dataset_name}")
        
    logger.info(f"Finished processing {dataset_name} for {lang}")

def create_combined_dataset(output_file: Path, langs: List[str]) -> None:
    """
    Create a combined dataset from processed files for training
    
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
    parser = argparse.ArgumentParser(description="Download and process datasets for Indic language summarization")
    
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
        choices=["te", "hi", "all"],
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
    
    # Download datasets
    if not args.process_only:
        for dataset_name in datasets:
            for lang in langs:
                try:
                    if lang in DATASET_CONFIGS[dataset_name]["languages"]:
                        download_dataset(dataset_name, lang, RAW_DATA_DIR)
                except Exception as e:
                    logger.error(f"Error downloading {dataset_name} for {lang}: {e}")
                    
    # Process datasets
    if not args.download_only:
        for dataset_name in datasets:
            for lang in langs:
                try:
                    if lang in DATASET_CONFIGS[dataset_name]["languages"]:
                        process_dataset(dataset_name, lang)
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