import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TOKENIZER_DIR = MODELS_DIR / "hf_smollm2-135m"

def analyze_source(name, hf_id, config_name, split, text_field, target_samples, total_docs_est, total_tokens_est):
    print(f"\nAnalyzing {name} ({hf_id})...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    
    ds = load_dataset(hf_id, config_name, split=split, streaming=True)
    
    lengths = []
    t0 = time.time()
    for idx, item in enumerate(ds):
        if idx >= target_samples:
            break
        text = item.get(text_field, "")
        if not text and isinstance(item, dict):
            # fallback find first text field
            for k, v in item.items():
                if isinstance(v, str) and len(v) > 50:
                    text = v
                    break
        if not text:
            continue
            
        # Fast token count estimation or exact tokenization
        # len(tokenizer.encode) on batches or exact
        tok_count = len(tokenizer.encode(text, truncation=False))
        lengths.append(tok_count)
        if (idx + 1) % 500 == 0 or (idx + 1) == target_samples:
            print(f"  Processed {idx + 1}/{target_samples} docs ({time.time() - t0:.1f}s)...")
            
    lengths = np.array(lengths)
    n_sampled = len(lengths)
    total_tokens_sampled = lengths.sum()
    mean_len = float(lengths.mean())
    median_len = float(np.median(lengths))
    p90_len = float(np.percentile(lengths, 90))
    p99_len = float(np.percentile(lengths, 99))
    max_len = int(lengths.max())
    
    # Bucket counts
    ge_2k_mask = lengths >= 2048
    ge_8k_mask = lengths >= 8192
    ge_32k_mask = lengths >= 32768
    
    pct_ge_2k = float(ge_2k_mask.mean() * 100)
    pct_ge_8k = float(ge_8k_mask.mean() * 100)
    pct_ge_32k = float(ge_32k_mask.mean() * 100)
    
    tok_pct_ge_2k = float(lengths[ge_2k_mask].sum() / total_tokens_sampled * 100) if total_tokens_sampled > 0 else 0
    tok_pct_ge_8k = float(lengths[ge_8k_mask].sum() / total_tokens_sampled * 100) if total_tokens_sampled > 0 else 0
    tok_pct_ge_32k = float(lengths[ge_32k_mask].sum() / total_tokens_sampled * 100) if total_tokens_sampled > 0 else 0
    
    # Extrapolated full dataset totals
    est_total_docs = total_docs_est
    est_total_tokens = total_tokens_est
    if est_total_tokens is None and est_total_docs is not None:
        est_total_tokens = est_total_docs * mean_len
    elif est_total_docs is None and est_total_tokens is not None:
        est_total_docs = int(est_total_tokens / mean_len)
        
    est_tok_ge_2k = (est_total_tokens * (tok_pct_ge_2k / 100.0)) if est_total_tokens else None
    est_tok_ge_8k = (est_total_tokens * (tok_pct_ge_8k / 100.0)) if est_total_tokens else None
    est_tok_ge_32k = (est_total_tokens * (tok_pct_ge_32k / 100.0)) if est_total_tokens else None

    return {
        "source": name,
        "hf_id": hf_id,
        "sampled_docs": n_sampled,
        "mean_tokens": round(mean_len, 1),
        "median_tokens": int(median_len),
        "p90_tokens": int(p90_len),
        "max_tokens": max_len,
        "docs_ge_2k_pct": round(pct_ge_2k, 2),
        "docs_ge_8k_pct": round(pct_ge_8k, 2),
        "docs_ge_32k_pct": round(pct_ge_32k, 2),
        "tok_pct_ge_2k": round(tok_pct_ge_2k, 1),
        "tok_pct_ge_8k": round(tok_pct_ge_8k, 1),
        "tok_pct_ge_32k": round(tok_pct_ge_32k, 1),
        "est_total_docs": est_total_docs,
        "est_total_tokens": est_total_tokens,
        "est_tokens_ge_2k": est_tok_ge_2k,
        "est_tokens_ge_8k": est_tok_ge_8k,
        "est_tokens_ge_32k": est_tok_ge_32k,
    }

def main():
    sources = [
        {
            "name": "TinyStories",
            "hf_id": "roneneldan/TinyStories",
            "config_name": "default",
            "split": "train",
            "text_field": "text",
            "target_samples": 2000,
            "total_docs_est": 2119719,      # ~2.1M stories
            "total_tokens_est": 460_000_000, # ~460M tokens
        },
        {
            "name": "Cosmopedia v2",
            "hf_id": "HuggingFaceTB/cosmopedia-v2",
            "config_name": "cosmopedia-v2",
            "split": "train",
            "text_field": "text",
            "target_samples": 1500,
            "total_docs_est": 39000000,
            "total_tokens_est": 28_000_000_000, # ~28B tokens
        },
        {
            "name": "FineWeb-Edu (10BT)",
            "hf_id": "HuggingFaceFW/fineweb-edu",
            "config_name": "sample-10BT",
            "split": "train",
            "text_field": "text",
            "target_samples": 1500,
            "total_docs_est": 9673924,
            "total_tokens_est": 10_000_000_000, # 10B tokens
        },
        {
            "name": "Wikipedia (en)",
            "hf_id": "wikimedia/wikipedia",
            "config_name": "20231101.en",
            "split": "train",
            "text_field": "text",
            "target_samples": 1500,
            "total_docs_est": 6407814,
            "total_tokens_est": 4_200_000_000, # ~4.2B tokens
        },
        {
            "name": "PG-19 (Books)",
            "hf_id": "emozilla/pg19",
            "config_name": "default",
            "split": "train",
            "text_field": "text",
            "target_samples": 150,
            "total_docs_est": 28752,
            "total_tokens_est": 1_970_000_000, # ~1.97B tokens in 28k books
        },
        {
            "name": "arXiv Papers",
            "hf_id": "ccdv/arxiv-summarization",
            "config_name": "section",
            "split": "train",
            "text_field": "article",
            "target_samples": 400,
            "total_docs_est": 203037,
            "total_tokens_est": 1_400_000_000, # ~1.4B tokens in 203k papers
        },
    ]

    results = []
    for s in sources:
        res = analyze_source(
            s["name"], s["hf_id"], s["config_name"], s["split"],
            s["text_field"], s["target_samples"], s["total_docs_est"], s["total_tokens_est"]
        )
        results.append(res)

    df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "length_distribution.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved length distribution analysis to {out_csv}")

if __name__ == "__main__":
    main()
