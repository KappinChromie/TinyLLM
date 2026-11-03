"""
Train Custom 8k SentencePiece BPE Tokenizer
- Mixture: ~90% Stage A (Cosmopedia v2, FineWeb-Edu, TinyStories) + ~10% Long-form (Wikipedia, PG-19)
- Vocab size: 8,192
- byte_fallback: True
- Reports compression ratio (tokens/char, bytes/token) across all candidate sources.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm
from transformers import LlamaTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
TOKENIZER_DIR = MODELS_DIR / "tokenizer_8k"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CORPUS_TXT = DATA_DIR / "tokenizer_train_corpus.txt"

def build_tokenizer_corpus():
    print("Building representative corpus for tokenizer training...")
    t0 = time.time()
    sampled_lines = []
    
    # 1. TinyStories (~15% of Stage A share, ~100k stories)
    print("  Sampling TinyStories...")
    df_ts = pd.read_parquet(RAW_DIR / "tinystories" / "train.parquet")
    sample_ts = df_ts.sample(min(80000, len(df_ts)), random_state=42)["text"]
    for t in sample_ts:
        for line in t.split("\n"):
            line = line.strip()
            if len(line) > 20:
                sampled_lines.append(line)
                
    # 2. Cosmopedia v2 (~40% share)
    print("  Sampling Cosmopedia v2...")
    cosmo_files = list((RAW_DIR / "cosmopedia").glob("**/*.parquet"))
    for f in cosmo_files[:3]:
        df = pd.read_parquet(f, columns=["text"])
        sample = df.sample(min(25000, len(df)), random_state=42)["text"]
        for t in sample:
            for line in t.split("\n"):
                line = line.strip()
                if len(line) > 30:
                    sampled_lines.append(line)
                    
    # 3. FineWeb-Edu (~35% share)
    print("  Sampling FineWeb-Edu...")
    fineweb_files = list((RAW_DIR / "fineweb_edu").glob("**/*.parquet"))
    for f in fineweb_files[:2]:
        df = pd.read_parquet(f, columns=["text"])
        sample = df.sample(min(25000, len(df)), random_state=42)["text"]
        for t in sample:
            for line in t.split("\n"):
                line = line.strip()
                if len(line) > 30:
                    sampled_lines.append(line)
                    
    # 4. Wikipedia (~5% long-form share)
    print("  Sampling Wikipedia...")
    wiki_files = list((RAW_DIR / "wikipedia").glob("**/*.parquet"))
    for f in wiki_files:
        if "00040" in str(f):
            continue # Skip neutral eval set!
        df = pd.read_parquet(f, columns=["text"])
        sample = df.sample(min(8000, len(df)), random_state=42)["text"]
        for t in sample:
            for line in t.split("\n"):
                line = line.strip()
                if len(line) > 40:
                    sampled_lines.append(line)
                    
    # 5. PG-19 Books (~5% long-form share)
    print("  Sampling PG-19 Books...")
    if (RAW_DIR / "pg19" / "train.parquet").exists():
        df_pg = pd.read_parquet(RAW_DIR / "pg19" / "train.parquet")
        for t in df_pg["text"][:80]:
            lines = [l.strip() for l in t.split("\n") if len(l.strip()) > 40]
            sampled_lines.extend(lines[:300])
            
    print(f"Total lines collected: {len(sampled_lines):,} ({time.time()-t0:.1f}s)")
    with open(CORPUS_TXT, "w", encoding="utf-8") as f:
        for line in sampled_lines:
            f.write(line + "\n")
    print(f"Saved corpus to {CORPUS_TXT} ({CORPUS_TXT.stat().st_size / (1024*1024):.1f} MB)")

def train_sentencepiece():
    print("\nTraining 8k SentencePiece BPE tokenizer...")
    t0 = time.time()
    model_prefix = str(TOKENIZER_DIR / "spm_8k")
    
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_TXT),
        model_prefix=model_prefix,
        vocab_size=8192,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=[],
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=True,
        input_sentence_size=2000000,
        shuffle_input_sentence=True
    )
    print(f"SentencePiece model trained ({time.time()-t0:.1f}s)")
    
    # Save standard tokenizer.model copy for GGUF / llama.cpp compatibility
    import shutil
    shutil.copy(TOKENIZER_DIR / "spm_8k.model", TOKENIZER_DIR / "tokenizer.model")
    
    # Export HuggingFace LlamaTokenizerFast
    spm_model_file = str(TOKENIZER_DIR / "tokenizer.model")
    hf_tok = LlamaTokenizerFast(
        vocab_file=spm_model_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>"
    )
    hf_tok.save_pretrained(str(TOKENIZER_DIR))
    print(f"Saved Hugging Face tokenizer to {TOKENIZER_DIR}")

def evaluate_compression():
    print("\nEvaluating tokenizer compression efficiency across sources...")
    spm_model_file = str(TOKENIZER_DIR / "spm_8k.model")
    sp = spm.SentencePieceProcessor(model_file=spm_model_file)
    
    sources = {
        "TinyStories": (RAW_DIR / "tinystories" / "validation.parquet", "text"),
        "Cosmopedia v2": (list((RAW_DIR / "cosmopedia").glob("**/*.parquet"))[0], "text"),
        "FineWeb-Edu": (list((RAW_DIR / "fineweb_edu").glob("**/*.parquet"))[0], "text"),
        "Wikipedia (en)": (list((RAW_DIR / "wikipedia").glob("**/*.parquet"))[0], "text"),
        "PG-19 (Books)": (RAW_DIR / "pg19" / "validation.parquet", "text"),
    }
    
    records = []
    for name, (path, text_field) in sources.items():
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        sample = df[text_field].dropna().sample(min(200, len(df)), random_state=123)
        
        total_chars = 0
        total_bytes = 0
        total_tokens = 0
        for doc in sample:
            doc_str = str(doc)
            doc_bytes = len(doc_str.encode("utf-8"))
            toks = len(sp.encode(doc_str))
            total_chars += len(doc_str)
            total_bytes += doc_bytes
            total_tokens += toks
            
        tok_per_char = total_tokens / max(1, total_chars)
        char_per_tok = total_chars / max(1, total_tokens)
        bytes_per_tok = total_bytes / max(1, total_tokens)
        
        records.append({
            "source": name,
            "sample_docs": len(sample),
            "tokens_per_char": round(tok_per_char, 4),
            "chars_per_token": round(char_per_tok, 3),
            "bytes_per_token": round(bytes_per_tok, 3)
        })
        print(f"  {name:<15s}: {char_per_tok:.3f} chars/tok | {bytes_per_tok:.3f} bytes/tok | {tok_per_char:.4f} tok/char")
        
    df_res = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "tokenizer_compression.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved compression report to {out_csv}")

if __name__ == "__main__":
    build_tokenizer_corpus()
    train_sentencepiece()
    evaluate_compression()
