"""
Comprehensive dataset verification script:
1. Inspects ccdv/arxiv-summarization (prints 5 samples, tests placeholders).
2. Re-samples ccdv/arxiv-summarization on a large sample (2,000 papers) for high-confidence >=32k stats.
3. Tests suolyer/pile_arxiv (clean arXiv from the Pile) on 2,000 papers.
4. Re-samples Wikipedia across 10 distributed shards (5,000 articles) to compute true mean/median/percentiles.
5. Measures PG-19 books on a large sample to compute direct token totals and >=32k fractions.
6. Measures 13-gram contamination between a 10M-token Wikipedia held-out set and FineWeb-Edu / Cosmopedia v2.
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TOKENIZER_DIR = ROOT / "models" / "hf_smollm2-135m"

def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def inspect_arxiv_samples():
    print("=" * 60)
    print("STEP 1: Inspecting ccdv/arxiv-summarization for placeholders")
    print("=" * 60)
    ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    samples = []
    for idx, item in enumerate(ds):
        if idx in [12, 45, 99, 150, 300]:
            samples.append(item["article"])
        if idx > 350:
            break
            
    for i, text in enumerate(samples):
        print(f"\n--- Sample {i+1} (first 400 chars) ---")
        print(text[:400].strip())
        has_xmath = "@xmath" in text
        has_xcite = "@xcite" in text
        count_xmath = text.count("@xmath")
        count_xcite = text.count("@xcite")
        print(f"Has @xmath? {has_xmath} (count: {count_xmath}) | Has @xcite? {has_xcite} (count: {count_xcite})")

def resample_arxiv(tokenizer):
    print("\n" + "=" * 60)
    print("STEP 2: Large-sample estimation for arXiv datasets")
    print("=" * 60)
    
    # 2A: ccdv/arxiv-summarization on 2000 papers
    print("Sampling 2,000 papers from ccdv/arxiv-summarization...")
    ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    lengths = []
    t0 = time.time()
    for idx, item in enumerate(ds):
        if idx >= 2000:
            break
        toks = len(tokenizer.encode(item["article"], truncation=False))
        lengths.append(toks)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/2000 papers ({time.time() - t0:.1f}s)...")
            
    lengths = np.array(lengths)
    n = len(lengths)
    k_32k = int((lengths >= 32768).sum())
    p_32k = k_32k / n
    ci_low, ci_high = wilson_ci(k_32k, n)
    
    print(f"ccdv/arxiv-summarization (N={n}):")
    print(f"  Mean tokens: {lengths.mean():.1f}, Median: {np.median(lengths):.1f}, p90: {np.percentile(lengths, 90):.1f}, Max: {lengths.max()}")
    print(f"  >=2k tokens: {(lengths >= 2048).mean()*100:.2f}%")
    print(f"  >=8k tokens: {(lengths >= 8192).mean()*100:.2f}%")
    print(f"  >=32k tokens: {p_32k*100:.2f}% ({k_32k}/{n} papers)")
    print(f"  95% CI for >=32k: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    
    # 2B: suolyer/pile_arxiv (clean arXiv) on 2000 papers
    print("\nSampling 2,000 papers from suolyer/pile_arxiv (validation)...")
    ds_clean = load_dataset("suolyer/pile_arxiv", split="validation", streaming=True)
    lengths_clean = []
    t0 = time.time()
    for idx, item in enumerate(ds_clean):
        if idx >= 2000:
            break
        toks = len(tokenizer.encode(item["text"], truncation=False))
        lengths_clean.append(toks)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/2000 clean papers ({time.time() - t0:.1f}s)...")
            
    lengths_clean = np.array(lengths_clean)
    n_c = len(lengths_clean)
    k_32k_c = int((lengths_clean >= 32768).sum())
    ci_low_c, ci_high_c = wilson_ci(k_32k_c, n_c)
    print(f"suolyer/pile_arxiv (N={n_c}):")
    print(f"  Mean tokens: {lengths_clean.mean():.1f}, Median: {np.median(lengths_clean):.1f}, p90: {np.percentile(lengths_clean, 90):.1f}, Max: {lengths_clean.max()}")
    print(f"  >=2k tokens: {(lengths_clean >= 2048).mean()*100:.2f}%")
    print(f"  >=8k tokens: {(lengths_clean >= 8192).mean()*100:.2f}%")
    print(f"  >=32k tokens: {(k_32k_c/n_c)*100:.2f}% ({k_32k_c}/{n_c} papers)")
    print(f"  95% CI for >=32k: [{ci_low_c*100:.2f}%, {ci_high_c*100:.2f}%]")
    
    return lengths, lengths_clean

def resample_wikipedia(tokenizer):
    print("\n" + "=" * 60)
    print("STEP 3: Multi-shard random sampling for Wikipedia (en)")
    print("=" * 60)
    
    # 10 shards distributed across the full 41-shard dataset
    target_shards = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    samples_per_shard = 500
    all_lengths = []
    t0 = time.time()
    
    for shard_idx in target_shards:
        shard_file = f"20231101.en/train-{shard_idx:05d}-of-00041.parquet"
        print(f"  Sampling {samples_per_shard} articles from shard {shard_idx} ({shard_file})...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", data_files=shard_file, streaming=True)
        shard_lens = []
        for i, item in enumerate(ds):
            if i >= samples_per_shard:
                break
            toks = len(tokenizer.encode(item["text"], truncation=False))
            shard_lens.append(toks)
        all_lengths.extend(shard_lens)
        print(f"    Shard {shard_idx} mean: {np.mean(shard_lens):.1f}, median: {np.median(shard_lens):.1f}")
        
    all_lengths = np.array(all_lengths)
    n = len(all_lengths)
    mean_val = float(all_lengths.mean())
    median_val = float(np.median(all_lengths))
    p90_val = float(np.percentile(all_lengths, 90))
    max_val = int(all_lengths.max())
    pct_2k = float((all_lengths >= 2048).mean() * 100)
    pct_8k = float((all_lengths >= 8192).mean() * 100)
    pct_32k = float((all_lengths >= 32768).mean() * 100)
    
    # Derivation of totals
    total_docs = 6407814
    derived_total_tokens = total_docs * mean_val
    tok_pct_2k = float(all_lengths[all_lengths >= 2048].sum() / all_lengths.sum() * 100)
    tok_pct_8k = float(all_lengths[all_lengths >= 8192].sum() / all_lengths.sum() * 100)
    tok_pct_32k = float(all_lengths[all_lengths >= 32768].sum() / all_lengths.sum() * 100) if (all_lengths >= 32768).any() else 0.0
    
    print(f"\nWikipedia Summary across {n} randomly distributed articles:")
    print(f"  Mean tokens: {mean_val:.1f}")
    print(f"  Median tokens: {median_val:.1f}")
    print(f"  p90 tokens: {p90_val:.1f}")
    print(f"  Max tokens: {max_val}")
    print(f"  Docs >= 2k: {pct_2k:.2f}% | Tokens in >=2k: {tok_pct_2k:.2f}%")
    print(f"  Docs >= 8k: {pct_8k:.2f}% | Tokens in >=8k: {tok_pct_8k:.2f}%")
    print(f"  Docs >= 32k: {pct_32k:.2f}%")
    print(f"  Derived Total Tokens = 6,407,814 docs * {mean_val:.1f} mean = {derived_total_tokens/1e9:.2f} Billion tokens")
    
    return all_lengths, derived_total_tokens

def resample_pg19(tokenizer):
    print("\n" + "=" * 60)
    print("STEP 4: PG-19 Direct Token Count and Length Verification")
    print("=" * 60)
    
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    lengths = []
    t0 = time.time()
    sample_target = 300
    for idx, item in enumerate(ds):
        if idx >= sample_target:
            break
        toks = len(tokenizer.encode(item["text"], truncation=False))
        lengths.append(toks)
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{sample_target} books ({time.time() - t0:.1f}s)...")
            
    lengths = np.array(lengths)
    n = len(lengths)
    mean_val = float(lengths.mean())
    median_val = float(np.median(lengths))
    p90_val = float(np.percentile(lengths, 90))
    max_val = int(lengths.max())
    pct_2k = float((lengths >= 2048).mean() * 100)
    pct_8k = float((lengths >= 8192).mean() * 100)
    pct_32k = float((lengths >= 32768).mean() * 100)
    tok_pct_32k = float(lengths[lengths >= 32768].sum() / lengths.sum() * 100)
    
    total_books = 28752
    derived_total_tokens = total_books * mean_val
    derived_tokens_ge_32k = derived_total_tokens * (tok_pct_32k / 100.0)
    
    print(f"\nPG-19 Summary across {n} books:")
    print(f"  Mean tokens: {mean_val:.1f}")
    print(f"  Median tokens: {median_val:.1f}")
    print(f"  p90 tokens: {p90_val:.1f}")
    print(f"  Max tokens: {max_val}")
    print(f"  Books >= 2k: {pct_2k:.2f}%")
    print(f"  Books >= 8k: {pct_8k:.2f}%")
    print(f"  Books >= 32k: {pct_32k:.2f}% ({int((lengths >= 32768).sum())}/{n} books)")
    print(f"  Tokens in >=32k: {tok_pct_32k:.2f}%")
    print(f"  Derived Total Tokens = {total_books} books * {mean_val:.1f} mean = {derived_total_tokens/1e9:.2f} Billion tokens")
    print(f"  Derived Tokens >=32k = {derived_tokens_ge_32k/1e9:.2f} Billion tokens")
    
    return lengths, derived_total_tokens, derived_tokens_ge_32k

def check_neutral_contamination():
    print("\n" + "=" * 60)
    print("STEP 5: 13-gram Contamination Check: Wikipedia vs Phase 5 Training Sets")
    print("=" * 60)
    
    # Extract 10M tokens (~12,000 articles) from held-out Wikipedia shard 40
    print("Extracting 10M token candidate neutral set from Wikipedia shard 40...")
    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", 
                           data_files="20231101.en/train-00040-of-00041.parquet", streaming=True)
    
    wiki_ngrams = set()
    wiki_doc_count = 0
    wiki_word_count = 0
    target_words = 8_000_000  # ~10M subword tokens
    
    for item in ds_wiki:
        text = item["text"].lower()
        words = text.split()
        if len(words) < 20:
            continue
        wiki_doc_count += 1
        wiki_word_count += len(words)
        # build 13-grams
        for i in range(len(words) - 12):
            wiki_ngrams.add(" ".join(words[i:i+13]))
        if wiki_word_count >= target_words or wiki_doc_count >= 15000:
            break
            
    print(f"  Neutral Wikipedia set: {wiki_doc_count} docs, {wiki_word_count} words, {len(wiki_ngrams):,} unique 13-grams")
    
    # Test contamination with FineWeb-Edu (sample 2,000 docs)
    print("Checking overlap with FineWeb-Edu (2,000 sample documents)...")
    ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    fw_total_13grams = 0
    fw_hits = 0
    fw_docs_contaminated = 0
    
    for idx, item in enumerate(ds_fineweb):
        if idx >= 2000:
            break
        text = item["text"].lower()
        words = text.split()
        doc_hit = False
        for i in range(len(words) - 12):
            fw_total_13grams += 1
            gram = " ".join(words[i:i+13])
            if gram in wiki_ngrams:
                fw_hits += 1
                doc_hit = True
        if doc_hit:
            fw_docs_contaminated += 1
            
    fw_gram_rate = (fw_hits / max(1, fw_total_13grams)) * 100
    fw_doc_rate = (fw_docs_contaminated / 2000) * 100
    print(f"  FineWeb-Edu: {fw_hits}/{fw_total_13grams:,} 13-grams matched ({fw_gram_rate:.4f}%) | {fw_docs_contaminated}/2000 docs with >=1 match ({fw_doc_rate:.2f}%)")
    
    # Test contamination with Cosmopedia v2 (sample 2,000 docs)
    print("Checking overlap with Cosmopedia v2 (2,000 sample documents)...")
    ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)
    cosmo_total_13grams = 0
    cosmo_hits = 0
    cosmo_docs_contaminated = 0
    
    for idx, item in enumerate(ds_cosmo):
        if idx >= 2000:
            break
        text = item["text"].lower()
        words = text.split()
        doc_hit = False
        for i in range(len(words) - 12):
            cosmo_total_13grams += 1
            gram = " ".join(words[i:i+13])
            if gram in wiki_ngrams:
                cosmo_hits += 1
                doc_hit = True
        if doc_hit:
            cosmo_docs_contaminated += 1
            
    cosmo_gram_rate = (cosmo_hits / max(1, cosmo_total_13grams)) * 100
    cosmo_doc_rate = (cosmo_docs_contaminated / 2000) * 100
    print(f"  Cosmopedia v2: {cosmo_hits}/{cosmo_total_13grams:,} 13-grams matched ({cosmo_gram_rate:.4f}%) | {cosmo_docs_contaminated}/2000 docs with >=1 match ({cosmo_doc_rate:.2f}%)")
    
    return {
        "wiki_docs": wiki_doc_count,
        "wiki_words": wiki_word_count,
        "fineweb_gram_rate": fw_gram_rate,
        "fineweb_doc_rate": fw_doc_rate,
        "cosmo_gram_rate": cosmo_gram_rate,
        "cosmo_doc_rate": cosmo_doc_rate
    }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    inspect_arxiv_samples()
    arxiv_lens, arxiv_clean_lens = resample_arxiv(tokenizer)
    wiki_lens, wiki_derived_tokens = resample_wikipedia(tokenizer)
    pg19_lens, pg19_derived_tokens, pg19_tokens_32k = resample_pg19(tokenizer)
    contam = check_neutral_contamination()
    print("\nALL DATA VERIFICATION CHECKS COMPLETED.")
