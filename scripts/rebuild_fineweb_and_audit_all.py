"""
Rebuild FineWeb-Edu Long Shards & Complete Document Overlap Audit
1. Rebuilds train_fineweb_edu_long.bin and val_fineweb_edu_long.bin strictly respecting
   the document-level train/val partition of the raw FineWeb-Edu parquets:
   - train partition: rows 0 to int(N * 0.95) - 1 (same docs as train_fineweb_edu.bin)
   - val partition: rows int(N * 0.95) to N - 1 (same docs as val_fineweb_edu.bin)
2. Audits document overlap across ALL dataset pairs:
   - val_fineweb_edu_long vs train_fineweb_edu
   - val_fineweb_edu_long vs train_fineweb_edu_long
   - val_fineweb_edu vs train_fineweb_edu
   - val_cosmopedia vs train_cosmopedia
   - val_wikipedia_neutral vs train_wikipedia_long
   - val_wikipedia_long vs train_wikipedia_long
   - val_pg19_long vs train_pg19_long
   - val_tinystories vs train_tinystories
3. Updates data/manifest.json and writes results/document_overlap_report.txt.
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

def doc_hash(text):
    # 64-bit hash of text for ultra-fast set membership
    return hashlib.sha256(text.encode("utf-8")).digest()[:8]

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

def rebuild_fineweb_long(sp):
    print("=" * 75, flush=True)
    print("  REBUILDING FINEWEB-EDU >= 8K (STRICT TRAIN / VAL PARTITIONS)", flush=True)
    print("=" * 75, flush=True)
    t0 = time.time()
    
    fw_files = sorted(list((RAW_DIR / "fineweb_edu").glob("**/*.parquet")))
    print(f"Found FineWeb-Edu parquet files: {[f.name for f in fw_files]}", flush=True)
    
    train_writer = BinaryShardWriter(SHARDS_DIR / "train_fineweb_edu_long.bin")
    val_writer = BinaryShardWriter(SHARDS_DIR / "val_fineweb_edu_long.bin")
    
    total_train_scanned = 0
    total_val_scanned = 0
    train_long_docs = 0
    val_long_docs = 0
    
    val_long_hashes = set()
    train_long_hashes = set()
    
    for fpath in fw_files:
        pf = pq.ParquetFile(fpath)
        total_rows = pf.metadata.num_rows
        n_train = int(total_rows * 0.95)
        print(f"  Processing {fpath.name}: {total_rows:,} rows (Train: 0..{n_train:,}, Val: {n_train:,}..{total_rows:,})", flush=True)
        
        row_offset = 0
        for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
            texts = batch.column(0).to_pylist()
            batch_len = len(texts)
            
            # Determine which rows in this batch are train vs val
            batch_start = row_offset
            batch_end = row_offset + batch_len
            
            # Sub-slice train rows
            if batch_start < n_train:
                train_slice_end = min(batch_end, n_train) - batch_start
                train_texts = texts[:train_slice_end]
                total_train_scanned += len(train_texts)
                
                # Filter long docs
                candidates = [t for t in train_texts if t and len(t) > 20000]
                if candidates:
                    encoded = sp.encode(candidates, num_threads=16)
                    long_texts = [c for c, toks in zip(candidates, encoded) if len(toks) >= 8192]
                    if long_texts:
                        train_long_docs += len(long_texts)
                        train_writer.write_docs(long_texts, sp, batch_size=50)
                        for t in long_texts:
                            train_long_hashes.add(doc_hash(t))
                            
            # Sub-slice val rows
            if batch_end > n_train:
                val_slice_start = max(0, n_train - batch_start)
                val_texts = texts[val_slice_start:]
                total_val_scanned += len(val_texts)
                
                candidates = [t for t in val_texts if t and len(t) > 20000]
                if candidates:
                    encoded = sp.encode(candidates, num_threads=16)
                    long_texts = [c for c, toks in zip(candidates, encoded) if len(toks) >= 8192]
                    if long_texts:
                        val_long_docs += len(long_texts)
                        val_writer.write_docs(long_texts, sp, batch_size=50)
                        for t in long_texts:
                            val_long_hashes.add(doc_hash(t))
                            
            row_offset += batch_len
            del texts, batch
            gc.collect()
            
    train_manifest = train_writer.close()
    val_manifest = val_writer.close()
    
    leakage = len(val_long_hashes.intersection(train_long_hashes))
    
    print(f"\nRebuild Finished in {time.time()-t0:.1f}s:", flush=True)
    print(f"  Train scanned: {total_train_scanned:,} docs -> Found >= 8k: {train_long_docs:,} docs", flush=True)
    print(f"  Val scanned: {total_val_scanned:,} docs -> Found >= 8k: {val_long_docs:,} docs", flush=True)
    print(f"  train_fineweb_edu_long.bin: {train_manifest['tokens']:,} tokens, {train_manifest['docs']:,} docs ({train_manifest['size_bytes']/(1024*1024):.1f} MB)", flush=True)
    print(f"  val_fineweb_edu_long.bin: {val_manifest['tokens']:,} tokens, {val_manifest['docs']:,} docs ({val_manifest['size_bytes']/(1024*1024):.1f} MB)", flush=True)
    print(f"  Direct Hash Leakage (val_long vs train_long): {leakage} documents", flush=True)
    
    # Update manifest
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    manifest["train_fineweb_edu_long"] = train_manifest
    manifest["val_fineweb_edu_long"] = val_manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated data/manifest.json successfully.", flush=True)
    
    return val_long_hashes, train_manifest, val_manifest

def audit_all_overlaps(val_fineweb_long_hashes):
    print("\n" + "=" * 75, flush=True)
    print("  RUNNING COMPREHENSIVE DOCUMENT OVERLAP AUDIT (HASH-LEVEL)", flush=True)
    print("=" * 75, flush=True)
    
    results = []
    
    # 1. val_fineweb_edu_long vs train_fineweb_edu (the key leakage check!)
    print("\n[1/7] Auditing val_fineweb_edu_long vs train_fineweb_edu...", flush=True)
    t0 = time.time()
    fw_files = sorted(list((RAW_DIR / "fineweb_edu").glob("**/*.parquet")))
    leakage_count = 0
    train_fw_docs = 0
    for fpath in fw_files:
        pf = pq.ParquetFile(fpath)
        n_train = int(pf.metadata.num_rows * 0.95)
        row_idx = 0
        for batch in pf.iter_batches(batch_size=20000, columns=["text"]):
            texts = batch.column(0).to_pylist()
            if row_idx + len(texts) <= n_train:
                slice_texts = texts
            elif row_idx < n_train:
                slice_texts = texts[:n_train - row_idx]
            else:
                break
            train_fw_docs += len(slice_texts)
            for t in slice_texts:
                if t and len(t) > 20000:
                    if doc_hash(t) in val_fineweb_long_hashes:
                        leakage_count += 1
            row_idx += len(texts)
            if row_idx >= n_train:
                break
    results.append({
        "Val Set": "val_fineweb_edu_long",
        "Train Set": "train_fineweb_edu",
        "Val Docs": len(val_fineweb_long_hashes),
        "Train Docs Audited": train_fw_docs,
        "Overlap Count": leakage_count,
        "Status": "PASS (0 overlap)" if leakage_count == 0 else "FAIL"
    })
    print(f"  Overlap: {leakage_count} / {len(val_fineweb_long_hashes):,} ({time.time()-t0:.1f}s)", flush=True)

    # 2. val_fineweb_edu vs train_fineweb_edu
    print("\n[2/7] Auditing val_fineweb_edu vs train_fineweb_edu...", flush=True)
    t0 = time.time()
    val_fw_hashes = set()
    for fpath in fw_files:
        pf = pq.ParquetFile(fpath)
        n_train = int(pf.metadata.num_rows * 0.95)
        row_idx = 0
        for batch in pf.iter_batches(batch_size=20000, columns=["text"]):
            texts = batch.column(0).to_pylist()
            if row_idx + len(texts) > n_train:
                start_val = max(0, n_train - row_idx)
                for t in texts[start_val:]:
                    if t:
                        val_fw_hashes.add(doc_hash(t))
            row_idx += len(texts)
    # Check against train
    overlap_fw = 0
    for fpath in fw_files:
        pf = pq.ParquetFile(fpath)
        n_train = int(pf.metadata.num_rows * 0.95)
        row_idx = 0
        for batch in pf.iter_batches(batch_size=20000, columns=["text"]):
            texts = batch.column(0).to_pylist()
            if row_idx + len(texts) <= n_train:
                slice_texts = texts
            elif row_idx < n_train:
                slice_texts = texts[:n_train - row_idx]
            else:
                break
            for t in slice_texts:
                if t and doc_hash(t) in val_fw_hashes:
                    overlap_fw += 1
            row_idx += len(texts)
            if row_idx >= n_train:
                break
    results.append({
        "Val Set": "val_fineweb_edu",
        "Train Set": "train_fineweb_edu",
        "Val Docs": len(val_fw_hashes),
        "Train Docs Audited": train_fw_docs,
        "Overlap Count": overlap_fw,
        "Status": "PASS (0 overlap)" if overlap_fw == 0 else "FAIL"
    })
    print(f"  Overlap: {overlap_fw} / {len(val_fw_hashes):,} ({time.time()-t0:.1f}s)", flush=True)

    # 3. val_cosmopedia vs train_cosmopedia
    print("\n[3/7] Auditing val_cosmopedia vs train_cosmopedia...", flush=True)
    t0 = time.time()
    cosmo_files = sorted(list((RAW_DIR / "cosmopedia").glob("**/*.parquet")))
    val_cosmo_hashes = set()
    for f in cosmo_files:
        df = pd.read_parquet(f, columns=["text"])
        n_train = int(len(df) * 0.95)
        for t in df["text"].iloc[n_train:]:
            if t:
                val_cosmo_hashes.add(doc_hash(t))
        del df
        gc.collect()
    overlap_cosmo = 0
    train_cosmo_docs = 0
    for f in cosmo_files:
        df = pd.read_parquet(f, columns=["text"])
        n_train = int(len(df) * 0.95)
        train_cosmo_docs += n_train
        for t in df["text"].iloc[:n_train]:
            if t and doc_hash(t) in val_cosmo_hashes:
                overlap_cosmo += 1
        del df
        gc.collect()
    results.append({
        "Val Set": "val_cosmopedia",
        "Train Set": "train_cosmopedia",
        "Val Docs": len(val_cosmo_hashes),
        "Train Docs Audited": train_cosmo_docs,
        "Overlap Count": overlap_cosmo,
        "Status": "PASS (0 overlap)" if overlap_cosmo == 0 else "FAIL"
    })
    print(f"  Overlap: {overlap_cosmo} / {len(val_cosmo_hashes):,} ({time.time()-t0:.1f}s)", flush=True)

    # 4. val_wikipedia_neutral vs train_wikipedia_long
    print("\n[4/7] Auditing val_wikipedia_neutral (Shard 40) vs train_wikipedia_long...", flush=True)
    t0 = time.time()
    wiki_files = sorted(list((RAW_DIR / "wikipedia").glob("**/*.parquet")))
    shard_40 = [f for f in wiki_files if "00040" in str(f)][0]
    df_neutral = pd.read_parquet(shard_40, columns=["text"])
    neutral_docs = df_neutral["text"].dropna().iloc[:12000].tolist()
    val_neutral_hashes = {doc_hash(t) for t in neutral_docs}
    del df_neutral
    gc.collect()
    
    other_wiki = [f for f in wiki_files if "00040" not in str(f)]
    train_wiki_docs = 0
    overlap_wiki_neutral = 0
    for f in other_wiki:
        df = pd.read_parquet(f, columns=["text"])
        long_texts = [t for t in df["text"].dropna() if len(t) > 15000]
        n_t = int(len(long_texts) * 0.95)
        train_texts = long_texts[:n_t]
        train_wiki_docs += len(train_texts)
        for t in train_texts:
            if doc_hash(t) in val_neutral_hashes:
                overlap_wiki_neutral += 1
        del df
        gc.collect()
    results.append({
        "Val Set": "val_wikipedia_neutral",
        "Train Set": "train_wikipedia_long",
        "Val Docs": len(val_neutral_hashes),
        "Train Docs Audited": train_wiki_docs,
        "Overlap Count": overlap_wiki_neutral,
        "Status": "PASS (0 overlap)" if overlap_wiki_neutral == 0 else "FAIL"
    })
    print(f"  Overlap: {overlap_wiki_neutral} / {len(val_neutral_hashes):,} ({time.time()-t0:.1f}s)", flush=True)

    # 5. val_wikipedia_long vs train_wikipedia_long
    print("\n[5/7] Auditing val_wikipedia_long vs train_wikipedia_long...", flush=True)
    t0 = time.time()
    val_wiki_long_hashes = set()
    for f in other_wiki:
        df = pd.read_parquet(f, columns=["text"])
        long_texts = [t for t in df["text"].dropna() if len(t) > 15000]
        n_t = int(len(long_texts) * 0.95)
        val_texts = long_texts[n_t:]
        for t in val_texts:
            val_wiki_long_hashes.add(doc_hash(t))
        del df
        gc.collect()
    overlap_wiki_long = 0
    for f in other_wiki:
        df = pd.read_parquet(f, columns=["text"])
        long_texts = [t for t in df["text"].dropna() if len(t) > 15000]
        n_t = int(len(long_texts) * 0.95)
        train_texts = long_texts[:n_t]
        for t in train_texts:
            if doc_hash(t) in val_wiki_long_hashes:
                overlap_wiki_long += 1
        del df
        gc.collect()
    results.append({
        "Val Set": "val_wikipedia_long",
        "Train Set": "train_wikipedia_long",
        "Val Docs": len(val_wiki_long_hashes),
        "Train Docs Audited": train_wiki_docs,
        "Overlap Count": overlap_wiki_long,
        "Status": "PASS (0 overlap)" if overlap_wiki_long == 0 else "FAIL"
    })
    print(f"  Overlap: {overlap_wiki_long} / {len(val_wiki_long_hashes):,} ({time.time()-t0:.1f}s)", flush=True)

    # 6. val_pg19_long vs train_pg19_long
    print("\n[6/7] Auditing val_pg19_long vs train_pg19_long (by book text hash)...", flush=True)
    t0 = time.time()
    pg_files = sorted(list((RAW_DIR / "pg19").glob("**/*.parquet")))
    full_shards = [f for f in pg_files if "train-0000" in f.name]
    val_files = [f for f in pg_files if "validation" in f.name and f.stat().st_size > 5 * 1024 * 1024]
    test_files = [f for f in pg_files if "test" in f.name and f.stat().st_size > 15 * 1024 * 1024]
    all_books = []
    for f in full_shards + val_files + test_files:
        df = pd.read_parquet(f, columns=["text"])
        all_books.extend(df["text"].dropna().tolist())
        del df
        gc.collect()
    val_books = all_books[:100]
    train_books = all_books[100:]
    val_pg19_hashes = {doc_hash(b) for b in val_books}
    overlap_pg19 = sum(1 for b in train_books if doc_hash(b) in val_pg19_hashes)
    results.append({
        "Val Set": "val_pg19_long",
        "Train Set": "train_pg19_long",
        "Val Docs": len(val_pg19_hashes),
        "Train Docs Audited": len(train_books),
        "Overlap Count": overlap_pg19,
        "Status": "PASS (0 overlap)" if overlap_pg19 == 0 else "FAIL"
    })
    print(f"  Overlap: {overlap_pg19} / {len(val_pg19_hashes):,} ({time.time()-t0:.1f}s)", flush=True)

    # 7. val_tinystories vs train_tinystories
    print("\n[7/7] Auditing val_tinystories vs train_tinystories...", flush=True)
    t0 = time.time()
    df_ts_val = pd.read_parquet(RAW_DIR / "tinystories" / "validation.parquet", columns=["text"])
    val_ts_hashes = {doc_hash(t) for t in df_ts_val["text"].dropna()}
    df_ts_train = pd.read_parquet(RAW_DIR / "tinystories" / "train.parquet", columns=["text"])
    overlap_ts = sum(1 for t in df_ts_train["text"].dropna() if doc_hash(t) in val_ts_hashes)
    results.append({
        "Val Set": "val_tinystories",
        "Train Set": "train_tinystories",
        "Val Docs": len(val_ts_hashes),
        "Train Docs Audited": len(df_ts_train),
        "Overlap Count": overlap_ts,
        "Status": "PASS (0 overlap)" if overlap_ts == 0 else f"FLAG ({overlap_ts} overlap in upstream split)"
    })
    print(f"  Overlap: {overlap_ts} / {len(val_ts_hashes):,} ({time.time()-t0:.1f}s)", flush=True)
    del df_ts_val, df_ts_train
    gc.collect()

    # Save summary report
    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_DIR / "document_overlap_table.csv", index=False)
    
    report_text = "=" * 80 + "\n"
    report_text += " DOCUMENT-LEVEL TRAIN / VALIDATION OVERLAP AUDIT REPORT\n"
    report_text += "=" * 80 + "\n\n"
    report_text += df_res.to_string(index=False) + "\n\n"
    
    with open(RESULTS_DIR / "document_overlap_report.txt", "w") as f:
        f.write(report_text)
    
    print("\n" + report_text, flush=True)

def main():
    print("Loading 8k SentencePiece tokenizer...", flush=True)
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    
    val_long_hashes, train_manifest, val_manifest = rebuild_fineweb_long(sp)
    audit_all_overlaps(val_long_hashes)
    print("\nAll tasks completed successfully!", flush=True)

if __name__ == "__main__":
    main()
