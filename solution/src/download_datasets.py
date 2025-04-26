import os
from pathlib import Path
import requests
from tqdm import tqdm
import gzip
import shutil
import sys

def download_file(url, filename):
    """Download a file from a URL to a local path with progress bar"""
    try:
        # Using a browser-like User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as progress_bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                progress_bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def extract_gz(gz_path, output_path):
    """Extract a gzip file"""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"Error extracting {gz_path}: {str(e)}")
        return False

def download_indicnlp_datasets():
    # Base URL for AI4Bharat corpus from official release
    base_url = "https://github.com/AI4Bharat/indicnlp_corpus/releases/download/v1.0"
    
    # Create data directories
    base_dir = Path("d:/Personal/IITP/Projects/SLM/solution/data")
    raw_dir = base_dir / "raw"
    
    # Language codes and their directories
    lang_codes = {
        "telugu": "te",
        "hindi": "hi"
    }
    
    # Create directories
    for lang in lang_codes.keys():
        (raw_dir / lang).mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    print("\nDownloading AI4Bharat IndicNLP Corpus...")
    
    for lang, code in lang_codes.items():
        print(f"\nProcessing {lang.capitalize()} datasets...")
        lang_dir = raw_dir / lang
        
        # Files to download with their new names
        files = [
            (f"{code}/{code}.txt.gz", f"{code}.txt"),
            (f"{code}/{code}.vocab.gz", f"{code}.vocab")
        ]
        
        lang_success = True
        for remote_path, local_name in files:
            gz_path = lang_dir / f"{local_name}.gz"
            txt_path = lang_dir / local_name
            file_url = f"{base_url}/{remote_path}"
            
            print(f"Downloading {local_name}.gz...")
            if not download_file(file_url, gz_path):
                # Try alternative URL format
                alt_url = f"{base_url}/indicnlp-articles-{code}.txt.gz"
                if not download_file(alt_url, gz_path):
                    lang_success = False
                    break
            
            print(f"Extracting {local_name}.gz...")
            if not extract_gz(gz_path, txt_path):
                lang_success = False
                break
            
            # Clean up gz file
            try:
                os.remove(gz_path)
            except Exception as e:
                print(f"Warning: Could not remove {gz_path}: {str(e)}")
        
        if lang_success:
            print(f"\n{lang.capitalize()} Dataset Statistics:")
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    num_lines = sum(1 for _ in f)
                with open(lang_dir / f"{code}.vocab", 'r', encoding='utf-8') as f:
                    vocab_size = sum(1 for _ in f)
                print(f"Number of sentences: {num_lines:,}")
                print(f"Vocabulary size: {vocab_size:,}")
            except Exception as e:
                print(f"Error reading statistics: {str(e)}")
        else:
            print(f"Failed to process {lang} dataset completely")
    
    print("\nTrying to download additional resources...")
    # Try downloading parallel corpus
    samanantar_url = "https://storage.googleapis.com/samanantar-public/V0.3/te-hi.zip"
    parallel_dir = raw_dir / "parallel"
    parallel_dir.mkdir(exist_ok=True)
    parallel_path = parallel_dir / "te-hi.zip"
    
    print("Downloading Samanantar Telugu-Hindi parallel corpus...")
    if download_file(samanantar_url, parallel_path):
        print("Successfully downloaded parallel corpus!")
    else:
        print("Could not download parallel corpus, will proceed with monolingual data only")

if __name__ == "__main__":
    download_indicnlp_datasets()