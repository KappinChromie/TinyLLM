"""
Download script for TinyLLM training and evaluation slices.
Downloads:
1. roneneldan/TinyStories (train + validation) [already downloaded]
2. HuggingFaceTB/cosmopedia-v2 (shards 0-5) [already downloaded]
3. HuggingFaceFW/fineweb-edu (sample-10BT: 000_00000.parquet [done], 001_00000.parquet)
4. wikimedia/wikipedia (20231101.en: Shards 40, 0, 16)
5. emozilla/pg19 (train + validation + test books >= 32k)
"""

import os
import sys
import time
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_fineweb_edu_shard1():
    dest = RAW_DIR / "fineweb_edu"
    target_file = dest / "sample" / "10BT" / "001_00000.parquet"
    if target_file.exists():
        print("FineWeb-Edu shard 1 already exists. Skipping.")
        return
    print("\n--- Downloading FineWeb-Edu shard 1 (001_00000.parquet) ---")
    t0 = time.time()
    hf_hub_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        filename="sample/10BT/001_00000.parquet",
        repo_type="dataset",
        local_dir=dest
    )
    print(f"  FineWeb-Edu shard 1 downloaded ({time.time()-t0:.1f}s)")

def download_wikipedia():
    dest = RAW_DIR / "wikipedia"
    dest.mkdir(exist_ok=True)
    print("\n--- Downloading Wikipedia shards (neutral eval + long articles) ---")
    t0 = time.time()
    shards = [
        "20231101.en/train-00040-of-00041.parquet", # Neutral eval set
        "20231101.en/train-00000-of-00041.parquet", # Shard 0 (long articles)
        "20231101.en/train-00016-of-00041.parquet", # Shard 16 (long articles)
    ]
    for s in shards:
        fname = s.split("/")[-1]
        target_file = dest / "20231101.en" / fname
        if target_file.exists():
            print(f"  {fname} already exists. Skipping.")
            continue
        print(f"  Downloading {fname}...")
        hf_hub_download(
            repo_id="wikimedia/wikipedia",
            filename=s,
            repo_type="dataset",
            local_dir=dest
        )
    print(f"  Wikipedia shards downloaded ({time.time()-t0:.1f}s)")

def download_pg19():
    dest = RAW_DIR / "pg19"
    dest.mkdir(exist_ok=True)
    if (dest / "train.parquet").exists() and (dest / "validation.parquet").exists():
        print("PG-19 already downloaded. Skipping.")
        return
    print("\n--- Downloading PG-19 (Books) targeted shards ---")
    t0 = time.time()
    
    # Download validation and test parquets directly via hf_hub_download
    val_file = hf_hub_download(
        repo_id="emozilla/pg19",
        filename="data/validation-00000-of-00001-0f92e2337f79aeac.parquet",
        repo_type="dataset",
        local_dir=dest
    )
    test_file = hf_hub_download(
        repo_id="emozilla/pg19",
        filename="data/test-00000-of-00001-29a571947c0b5ccc.parquet",
        repo_type="dataset",
        local_dir=dest
    )
    # Download single train shard 0 (contains ~1,250 books, we only need 300)
    train_shard0 = hf_hub_download(
        repo_id="emozilla/pg19",
        filename="data/train-00000-of-00023-5263dc4323e881d2.parquet",
        repo_type="dataset",
        local_dir=dest
    )
    
    df_val = pd.read_parquet(val_file)
    df_test = pd.read_parquet(test_file)
    df_train_shard = pd.read_parquet(train_shard0)
    df_train = df_train_shard.iloc[:300].copy()
    
    df_val.to_parquet(dest / "validation.parquet", index=False)
    df_test.to_parquet(dest / "test.parquet", index=False)
    df_train.to_parquet(dest / "train.parquet", index=False)
    
    print(f"  PG-19 saved: {len(df_train)} train books, {len(df_val)} val books, {len(df_test)} test books ({time.time()-t0:.1f}s)")

def main():
    download_fineweb_edu_shard1()
    download_wikipedia()
    download_pg19()
    
    total_bytes = sum(f.stat().st_size for f in RAW_DIR.glob("**/*") if f.is_file())
    print(f"\nALL DOWNLOADS COMPLETE.")
    print(f"Total raw disk usage: {total_bytes / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
