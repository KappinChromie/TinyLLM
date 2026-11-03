"""
Process Expanded PG-19 and Build FineWeb-Edu >=8k Long Dataset
1. Collects all downloaded PG-19 shards (~5,000 books).
   - Holds out exactly 100 books for validation (val_pg19_long.bin).
   - Uses remaining ~4,900 books for training (train_pg19_long.bin).
   - Measures exact token counts and length distribution with custom 8k tokenizer.
2. Carves out documents >= 8,192 tokens from FineWeb-Edu:
   - Streams downloaded shards (000_00000.parquet and 001_00000.parquet).
   - Filters and tokenizes documents >= 8,192 tokens.
   - Splits 95% train (train_fineweb_edu_long.bin), 5% val (val_fineweb_edu_long.bin).
3. Updates data/manifest.json and reports repetition rates for Stages B and C.
"""

import os
import sys
import gc
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SHARDS_DIR = DATA_DIR / "shards"
SHARDS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = DATA_DIR / "manifest.json"
TOKENIZER_MODEL = ROOT / "models" / "tokenizer_8k" / "tokenizer.model"
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

    def write_docs(self, docs, sp, batch_size=50, num_threads=16):
        docs = [d for d in docs if d and isinstance(d, str)]
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            encoded_batch = sp.encode(batch, num_threads=num_threads)
            
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

def process_pg19(sp):
    print("\n" + "=" * 70, flush=True)
    print("  PROCESSING EXPANDED PG-19 (8K TOKENIZER)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    
    pg_files = sorted(list((RAW_DIR / "pg19").glob("**/*.parquet")))
    # Exclude small test/train stubs if full shard files are present
    full_shards = [f for f in pg_files if "train-0000" in f.name]
    val_files = [f for f in pg_files if "validation" in f.name and f.stat().st_size > 5 * 1024 * 1024]
    test_files = [f for f in pg_files if "test" in f.name and f.stat().st_size > 15 * 1024 * 1024]
    
    print(f"Found PG-19 parquet files: {len(full_shards)} train shards, {len(val_files)} val, {len(test_files)} test", flush=True)
    
    all_books = []
    for f in full_shards + val_files + test_files:
        print(f"  Reading {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...", flush=True)
        df = pd.read_parquet(f, columns=["text"])
        all_books.extend(df["text"].dropna().tolist())
        del df
        gc.collect()
        
    print(f"Total PG-19 books loaded: {len(all_books):,}", flush=True)
    
    # Validation split: hold out exactly 100 books
    val_books = all_books[:100]
    train_books = all_books[100:]
    
    print(f"Train books: {len(train_books):,} | Val books: {len(val_books):,}", flush=True)
    
    # Measure length distribution on sample/all books
    print("Computing length distribution with 8k SentencePiece tokenizer...", flush=True)
    lengths = []
    # Sample 500 books to compute stats fast
    sample_books = all_books[:500]
    for b in sample_books:
        toks = len(sp.encode(b, num_threads=16))
        lengths.append(toks)
        
    lengths = np.array(lengths)
    mean_tok = float(np.mean(lengths))
    median_tok = float(np.median(lengths))
    share_ge_8k = float(np.mean(lengths >= 8192) * 100)
    tokens_ge_8k = float(np.sum(lengths[lengths >= 8192]) / np.sum(lengths) * 100)
    share_ge_32k = float(np.mean(lengths >= 32768) * 100)
    tokens_ge_32k = float(np.sum(lengths[lengths >= 32768]) / np.sum(lengths) * 100)
    
    print(f"  Mean tokens/book: {mean_tok:,.1f}", flush=True)
    print(f"  Median tokens/book: {median_tok:,.1f}", flush=True)
    print(f"  Books >= 8k: {share_ge_8k:.2f}% of books | {tokens_ge_8k:.2f}% of tokens", flush=True)
    print(f"  Books >= 32k: {share_ge_32k:.2f}% of books | {tokens_ge_32k:.2f}% of tokens", flush=True)
    
    # Write train shard
    print("Writing train_pg19_long.bin...", flush=True)
    train_writer = BinaryShardWriter(SHARDS_DIR / "train_pg19_long.bin")
    train_writer.write_docs(train_books, sp, batch_size=50)
    train_manifest = train_writer.close()
    print(f"  train_pg19_long.bin: {train_manifest['tokens']:,} tokens, {train_manifest['docs']:,} books ({train_manifest['size_bytes']/(1024*1024):.1f} MB)", flush=True)
    
    # Write val shard
    print("Writing val_pg19_long.bin...", flush=True)
    val_writer = BinaryShardWriter(SHARDS_DIR / "val_pg19_long.bin")
    val_writer.write_docs(val_books, sp, batch_size=50)
    val_manifest = val_writer.close()
    print(f"  val_pg19_long.bin: {val_manifest['tokens']:,} tokens, {val_manifest['docs']:,} books ({val_manifest['size_bytes']/(1024*1024):.1f} MB)", flush=True)
    
    stats = {
        "total_books": len(all_books),
        "train_books": len(train_books),
        "val_books": len(val_books),
        "mean_tokens": round(mean_tok, 1),
        "median_tokens": round(median_tok, 1),
        "share_books_ge_8k": round(share_ge_8k, 2),
        "share_tokens_ge_8k": round(tokens_ge_8k, 2),
        "share_books_ge_32k": round(share_ge_32k, 2),
        "share_tokens_ge_32k": round(tokens_ge_32k, 2),
        "train_tokens": train_manifest["tokens"],
        "val_tokens": val_manifest["tokens"],
        "elapsed_sec": round(time.time() - t0, 1)
    }
    return train_manifest, val_manifest, stats

def process_fineweb_long(sp):
    print("\n" + "=" * 70, flush=True)
    print("  CARVING OUT FINEWEB-EDU >= 8K TOKENS", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    
    fw_files = sorted(list((RAW_DIR / "fineweb_edu").glob("**/*.parquet")))
    print(f"Found FineWeb-Edu parquet files: {[f.name for f in fw_files]}", flush=True)
    
    train_writer = BinaryShardWriter(SHARDS_DIR / "train_fineweb_edu_long.bin")
    val_writer = BinaryShardWriter(SHARDS_DIR / "val_fineweb_edu_long.bin")
    
    total_docs_scanned = 0
    long_docs_found = 0
    total_long_tokens = 0
    
    for fpath in fw_files:
        pf = pq.ParquetFile(fpath)
        print(f"  Scanning {fpath.name} ({pf.metadata.num_rows:,} rows)...", flush=True)
        
        for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
            texts = batch.column(0).to_pylist()
            total_docs_scanned += len(texts)
            
            # Fast filter: 8k tokens in 8k tokenizer is ~25,000 chars
            candidates = [t for t in texts if t and len(t) > 20000]
            if candidates:
                encoded = sp.encode(candidates, num_threads=16)
                long_texts = [c for c, toks in zip(candidates, encoded) if len(toks) >= 8192]
                
                if long_texts:
                    long_docs_found += len(long_texts)
                    # 95/5 split
                    n_train = int(len(long_texts) * 0.95)
                    train_writer.write_docs(long_texts[:n_train], sp, batch_size=50)
                    val_writer.write_docs(long_texts[n_train:], sp, batch_size=50)
                    
            del texts, batch
            gc.collect()
            
    train_manifest = train_writer.close()
    val_manifest = val_writer.close()
    
    print(f"\nFineWeb-Edu >= 8k Extraction Complete in {time.time()-t0:.1f}s:", flush=True)
    print(f"  Scanned: {total_docs_scanned:,} docs | Found >= 8k: {long_docs_found:,} docs ({long_docs_found/total_docs_scanned*100:.2f}%)", flush=True)
    print(f"  train_fineweb_edu_long.bin: {train_manifest['tokens']:,} tokens, {train_manifest['docs']:,} docs ({train_manifest['size_bytes']/(1024*1024):.1f} MB)", flush=True)
    print(f"  val_fineweb_edu_long.bin: {val_manifest['tokens']:,} tokens, {val_manifest['docs']:,} docs ({val_manifest['size_bytes']/(1024*1024):.1f} MB)", flush=True)
    
    return train_manifest, val_manifest, {
        "scanned_docs": total_docs_scanned,
        "long_docs": long_docs_found,
        "train_tokens": train_manifest["tokens"],
        "val_tokens": val_manifest["tokens"],
        "elapsed_sec": round(time.time() - t0, 1)
    }

def main():
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    
    # 1. Process PG-19
    pg_train_man, pg_val_man, pg_stats = process_pg19(sp)
    
    # 2. Process FineWeb-Edu >= 8k
    fw_train_man, fw_val_man, fw_stats = process_fineweb_long(sp)
    
    # 3. Update manifest.json
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
        
    manifest["train_pg19_long"] = pg_train_man
    manifest["val_pg19_long"] = pg_val_man
    manifest["train_fineweb_edu_long"] = fw_train_man
    manifest["val_fineweb_edu_long"] = fw_val_man
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nUpdated manifest saved to {MANIFEST_PATH}", flush=True)
    
    # Save statistics report
    report_path = RESULTS_DIR / "stage_b_c_data_audit.json"
    audit_report = {
        "pg19_stats": pg_stats,
        "fineweb_edu_long_stats": fw_stats,
        "stage_b_repetition": {
            "wikipedia_long_35pct": {
                "tokens_needed": 87_500_000,
                "tokens_available": manifest["train_wikipedia_long"]["tokens"],
                "repetition_factor": round(87_500_000 / manifest["train_wikipedia_long"]["tokens"], 4)
            },
            "fineweb_edu_long_25pct": {
                "tokens_needed": 62_500_000,
                "tokens_available": fw_train_man["tokens"],
                "repetition_factor": round(62_500_000 / fw_train_man["tokens"], 4)
            },
            "pg19_books_15pct": {
                "tokens_needed": 37_500_000,
                "tokens_available": pg_train_man["tokens"],
                "repetition_factor": round(37_500_000 / pg_train_man["tokens"], 4)
            },
            "short_replay_25pct": {
                "tokens_needed": 62_500_000,
                "tokens_available": manifest["train_cosmopedia"]["tokens"] + manifest["train_fineweb_edu"]["tokens"],
                "repetition_factor": round(62_500_000 / (manifest["train_cosmopedia"]["tokens"] + manifest["train_fineweb_edu"]["tokens"]), 4)
            }
        },
        "stage_c_repetition": {
            "pg19_books_75pct": {
                "tokens_needed": 86_000_000,
                "tokens_available": int(pg_train_man["tokens"] * (pg_stats["share_tokens_ge_32k"] / 100.0)),
                "repetition_factor": round(86_000_000 / (pg_train_man["tokens"] * (pg_stats["share_tokens_ge_32k"] / 100.0)), 4)
            },
            "short_replay_25pct": {
                "tokens_needed": 28_700_000,
                "tokens_available": manifest["train_cosmopedia"]["tokens"] + manifest["train_fineweb_edu"]["tokens"],
                "repetition_factor": round(28_700_000 / (manifest["train_cosmopedia"]["tokens"] + manifest["train_fineweb_edu"]["tokens"]), 4)
            }
        }
    }
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Audit report saved to {report_path}", flush=True)

if __name__ == "__main__":
    main()
