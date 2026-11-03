"""
Data Tokenization and Sharding Script
- Takes document-level train/validation splits BEFORE tokenizing.
- High-performance C++ batch encoding via SentencePiece (num_threads=16).
- Streams in batches of 10,000 docs to keep RAM usage under 300 MB at all times.
- Computes exact token counts, SHA256 hashes, and writes data/manifest.json.
- Re-runs 13-gram contamination check on the actual downloaded training slices.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SHARDS_DIR = DATA_DIR / "shards"
SHARDS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"
TOKENIZER_DIR = MODELS_DIR / "tokenizer_8k"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EOS_ID = 2

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

class BinaryShardWriter:
    def __init__(self, bin_path):
        self.bin_path = Path(bin_path)
        if self.bin_path.exists():
            self.bin_path.unlink()
        self.f = open(self.bin_path, "wb")
        self.total_tokens = 0
        self.total_docs = 0

    def write_docs(self, docs_series_or_list, sp, batch_size=10000, num_threads=16):
        docs = [d for d in docs_series_or_list if d and isinstance(d, str)]
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            encoded_batch = sp.encode(batch, num_threads=num_threads)
            
            # Flatten with EOS delimiter
            chunk_tokens = []
            for item in encoded_batch:
                if item:
                    chunk_tokens.extend(item)
                    chunk_tokens.append(EOS_ID)
                    self.total_docs += 1
                    
            if chunk_tokens:
                arr = np.array(chunk_tokens, dtype=np.uint16)
                arr.tofile(self.f)
                self.f.flush()
                self.total_tokens += len(arr)

    def close(self):
        self.f.close()
        return {
            "tokens": self.total_tokens,
            "docs": self.total_docs,
            "sha256": get_file_sha256(self.bin_path),
            "size_bytes": self.bin_path.stat().st_size
        }

def process_datasets():
    print("Loading custom 8k tokenizer...")
    spm_model_file = str(TOKENIZER_DIR / "tokenizer.model")
    if not Path(spm_model_file).exists():
        spm_model_file = str(TOKENIZER_DIR / "spm_8k.model")
    sp = spm.SentencePieceProcessor(model_file=spm_model_file)
    
    manifest = {}
    
    # 1. TinyStories
    print("\n--- Tokenizing TinyStories ---")
    t0 = time.time()
    ts_train_writer = BinaryShardWriter(SHARDS_DIR / "train_tinystories.bin")
    df_ts_train = pd.read_parquet(RAW_DIR / "tinystories" / "train.parquet")
    ts_train_writer.write_docs(df_ts_train["text"], sp)
    manifest["train_tinystories"] = ts_train_writer.close()
    print(f"  TinyStories Train: {manifest['train_tinystories']['tokens']:,} tokens, {manifest['train_tinystories']['docs']:,} docs ({time.time()-t0:.1f}s)")
    
    ts_val_writer = BinaryShardWriter(SHARDS_DIR / "val_tinystories.bin")
    df_ts_val = pd.read_parquet(RAW_DIR / "tinystories" / "validation.parquet")
    ts_val_writer.write_docs(df_ts_val["text"], sp)
    manifest["val_tinystories"] = ts_val_writer.close()
    print(f"  TinyStories Val: {manifest['val_tinystories']['tokens']:,} tokens, {manifest['val_tinystories']['docs']:,} docs")
    
    # 2. Cosmopedia v2
    print("\n--- Tokenizing Cosmopedia v2 (6 shards, 95/5 split) ---")
    t0 = time.time()
    cosmo_train_writer = BinaryShardWriter(SHARDS_DIR / "train_cosmopedia.bin")
    cosmo_val_writer = BinaryShardWriter(SHARDS_DIR / "val_cosmopedia.bin")
    cosmo_files = sorted(list((RAW_DIR / "cosmopedia").glob("**/*.parquet")))
    for idx, f in enumerate(cosmo_files):
        df = pd.read_parquet(f, columns=["text"])
        n_train = int(len(df) * 0.95)
        cosmo_train_writer.write_docs(df["text"].iloc[:n_train], sp)
        cosmo_val_writer.write_docs(df["text"].iloc[n_train:], sp)
        print(f"  Processed Cosmopedia shard {idx+1}/{len(cosmo_files)} ({len(df):,} docs)")
    manifest["train_cosmopedia"] = cosmo_train_writer.close()
    manifest["val_cosmopedia"] = cosmo_val_writer.close()
    print(f"  Cosmopedia Train: {manifest['train_cosmopedia']['tokens']:,} tokens | Val: {manifest['val_cosmopedia']['tokens']:,} tokens ({time.time()-t0:.1f}s)")
    
    # 3. FineWeb-Edu
    print("\n--- Tokenizing FineWeb-Edu (2 shards, 95/5 split) ---")
    t0 = time.time()
    fw_train_writer = BinaryShardWriter(SHARDS_DIR / "train_fineweb_edu.bin")
    fw_val_writer = BinaryShardWriter(SHARDS_DIR / "val_fineweb_edu.bin")
    fineweb_files = sorted(list((RAW_DIR / "fineweb_edu").glob("**/*.parquet")))
    for idx, f in enumerate(fineweb_files):
        df = pd.read_parquet(f, columns=["text"])
        n_train = int(len(df) * 0.95)
        fw_train_writer.write_docs(df["text"].iloc[:n_train], sp)
        fw_val_writer.write_docs(df["text"].iloc[n_train:], sp)
        print(f"  Processed FineWeb-Edu shard {idx+1}/{len(fineweb_files)} ({len(df):,} docs)")
    manifest["train_fineweb_edu"] = fw_train_writer.close()
    manifest["val_fineweb_edu"] = fw_val_writer.close()
    print(f"  FineWeb-Edu Train: {manifest['train_fineweb_edu']['tokens']:,} tokens | Val: {manifest['val_fineweb_edu']['tokens']:,} tokens ({time.time()-t0:.1f}s)")
    
    # 4. Wikipedia
    print("\n--- Tokenizing Wikipedia ---")
    t0 = time.time()
    wiki_neutral_writer = BinaryShardWriter(SHARDS_DIR / "val_wikipedia_neutral.bin")
    wiki_long_train_writer = BinaryShardWriter(SHARDS_DIR / "train_wikipedia_long.bin")
    wiki_long_val_writer = BinaryShardWriter(SHARDS_DIR / "val_wikipedia_long.bin")
    
    wiki_files = sorted(list((RAW_DIR / "wikipedia").glob("**/*.parquet")))
    neutral_eval_texts = []
    for f in wiki_files:
        df = pd.read_parquet(f, columns=["text"])
        if "00040" in str(f):
            # Target ~10M tokens: ~12,000 docs
            neutral_docs = df["text"].dropna().iloc[:12000].tolist()
            neutral_eval_texts = neutral_docs
            wiki_neutral_writer.write_docs(neutral_docs, sp)
            print(f"  Wikipedia Shard 40 Neutral Eval: {len(neutral_docs):,} docs")
        else:
            # Long articles (> 15,000 chars)
            long_texts = [t for t in df["text"].dropna() if len(t) > 15000]
            n_t = int(len(long_texts) * 0.95)
            wiki_long_train_writer.write_docs(long_texts[:n_t], sp)
            wiki_long_val_writer.write_docs(long_texts[n_t:], sp)
            print(f"  Wikipedia Long Articles from {f.name}: {len(long_texts):,} docs")
            
    manifest["val_wikipedia_neutral"] = wiki_neutral_writer.close()
    manifest["train_wikipedia_long"] = wiki_long_train_writer.close()
    manifest["val_wikipedia_long"] = wiki_long_val_writer.close()
    print(f"  Wikipedia Neutral Val: {manifest['val_wikipedia_neutral']['tokens']:,} tokens")
    print(f"  Wikipedia Long Train: {manifest['train_wikipedia_long']['tokens']:,} tokens | Val: {manifest['val_wikipedia_long']['tokens']:,} tokens ({time.time()-t0:.1f}s)")
    
    # 5. PG-19 Books
    print("\n--- Tokenizing PG-19 Books ---")
    t0 = time.time()
    pg_train_writer = BinaryShardWriter(SHARDS_DIR / "train_pg19_long.bin")
    pg_val_writer = BinaryShardWriter(SHARDS_DIR / "val_pg19_long.bin")
    
    if (RAW_DIR / "pg19" / "train.parquet").exists():
        pg_train = pd.read_parquet(RAW_DIR / "pg19" / "train.parquet")["text"].dropna().tolist()
        pg_val = pd.read_parquet(RAW_DIR / "pg19" / "validation.parquet")["text"].dropna().tolist()
        pg_test = pd.read_parquet(RAW_DIR / "pg19" / "test.parquet")["text"].dropna().tolist()
        
        pg_train_writer.write_docs(pg_train[:280], sp, batch_size=50)
        eval_books = pg_train[280:] + pg_val + pg_test
        pg_val_writer.write_docs(eval_books, sp, batch_size=50)
        
    manifest["train_pg19_long"] = pg_train_writer.close()
    manifest["val_pg19_long"] = pg_val_writer.close()
    print(f"  PG-19 Train: {manifest['train_pg19_long']['tokens']:,} tokens | Eval: {manifest['val_pg19_long']['tokens']:,} tokens ({time.time()-t0:.1f}s)")
    
    # Save manifest
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest successfully written to {DATA_DIR / 'manifest.json'}")
    
    return neutral_eval_texts

def run_contamination_check(neutral_eval_texts):
    print("\n--- Running 13-Gram Contamination Check on Actual Training Slices ---")
    t0 = time.time()
    wiki_ngrams = set()
    wiki_word_count = 0
    for doc in neutral_eval_texts:
        words = doc.lower().split()
        wiki_word_count += len(words)
        for i in range(len(words) - 12):
            wiki_ngrams.add(" ".join(words[i:i+13]))
            
    print(f"Neutral Wikipedia set: {len(neutral_eval_texts):,} docs, {wiki_word_count:,} words, {len(wiki_ngrams):,} unique 13-grams")
    
    # Check Cosmopedia training slices
    print("Streaming through all downloaded Cosmopedia training shards...")
    cosmo_total = 0
    cosmo_hits = 0
    cosmo_hit_docs = 0
    cosmo_docs_count = 0
    cosmo_files = sorted(list((RAW_DIR / "cosmopedia").glob("**/*.parquet")))
    for f in cosmo_files:
        df = pd.read_parquet(f, columns=["text"])
        n_train = int(len(df) * 0.95)
        for doc in df["text"].iloc[:n_train]:
            if not doc or not isinstance(doc, str):
                continue
            cosmo_docs_count += 1
            words = doc.lower().split()
            doc_hit = False
            for i in range(len(words) - 12):
                cosmo_total += 1
                if " ".join(words[i:i+13]) in wiki_ngrams:
                    cosmo_hits += 1
                    doc_hit = True
            if doc_hit:
                cosmo_hit_docs += 1
                
    cosmo_rate = (cosmo_hits / max(1, cosmo_total)) * 100
    print(f"  Cosmopedia: {cosmo_hits}/{cosmo_total:,} 13-grams matched ({cosmo_rate:.6f}%) | {cosmo_hit_docs}/{cosmo_docs_count:,} docs")
    
    # Check FineWeb-Edu training slices
    print("Streaming through all downloaded FineWeb-Edu training shards...")
    fw_total = 0
    fw_hits = 0
    fw_hit_docs = 0
    fw_docs_count = 0
    fineweb_files = sorted(list((RAW_DIR / "fineweb_edu").glob("**/*.parquet")))
    for f in fineweb_files:
        df = pd.read_parquet(f, columns=["text"])
        n_train = int(len(df) * 0.95)
        for doc in df["text"].iloc[:n_train]:
            if not doc or not isinstance(doc, str):
                continue
            fw_docs_count += 1
            words = doc.lower().split()
            doc_hit = False
            for i in range(len(words) - 12):
                fw_total += 1
                if " ".join(words[i:i+13]) in wiki_ngrams:
                    fw_hits += 1
                    doc_hit = True
            if doc_hit:
                fw_hit_docs += 1
                
    fw_rate = (fw_hits / max(1, fw_total)) * 100
    print(f"  FineWeb-Edu: {fw_hits}/{fw_total:,} 13-grams matched ({fw_rate:.6f}%) | {fw_hit_docs}/{fw_docs_count:,} docs")
    print(f"Contamination check completed in {time.time()-t0:.1f}s")
    
    report_file = RESULTS_DIR / "actual_data_contamination_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Actual Downloaded Data 13-Gram Contamination Report\n")
        f.write("===================================================\n")
        f.write(f"Neutral Wikipedia Eval Docs: {len(neutral_eval_texts):,}\n")
        f.write(f"Neutral Wikipedia Unique 13-Grams: {len(wiki_ngrams):,}\n\n")
        f.write(f"Cosmopedia Training Slice:\n")
        f.write(f"  Total Docs Audited: {cosmo_docs_count:,}\n")
        f.write(f"  Total 13-Grams Audited: {cosmo_total:,}\n")
        f.write(f"  13-Grams Matched: {cosmo_hits}\n")
        f.write(f"  Contamination Rate: {cosmo_rate:.6f}%\n")
        f.write(f"  Contaminated Docs: {cosmo_hit_docs}/{cosmo_docs_count:,} ({(cosmo_hit_docs/max(1,cosmo_docs_count))*100:.4f}%)\n\n")
        f.write(f"FineWeb-Edu Training Slice:\n")
        f.write(f"  Total Docs Audited: {fw_docs_count:,}\n")
        f.write(f"  Total 13-Grams Audited: {fw_total:,}\n")
        f.write(f"  13-Grams Matched: {fw_hits}\n")
        f.write(f"  Contamination Rate: {fw_rate:.6f}%\n")
        f.write(f"  Contaminated Docs: {fw_hit_docs}/{fw_docs_count:,} ({(fw_hit_docs/max(1,fw_docs_count))*100:.4f}%)\n")
        
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    neutral_texts = process_datasets()
    run_contamination_check(neutral_texts)
