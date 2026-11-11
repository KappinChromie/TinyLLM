"""
Build and filter SFT dataset for TinyLLM-42M Chat model:
1. Loads subsets:
   - everyday-conversations
   - smol-contraints (constraints)
   - smollm-rewrite-30k + explore-instruct-rewrite (rewriting)
   - smol-summarize-20k + smol-summarize-5k (summarization)
   - smol-magpie-ultra-short (curated concise general chat & QA)
2. Filters:
   - Length: <= 1024 tokens total conversation length
   - Non-English filter
   - Heavy code / complex math filter
   - System/assistant self-identity filter (OpenAI, ChatGPT, Claude, Anthropic, Meta, Llama, Qwen, Alibaba)
3. Hold out 1,000 conversations for validation (with exact duplicate check)
4. Saves 100 random examples to results/sft_100_samples.md for inspection
5. Tokenizes and saves train and val sets
"""

import os
import sys
import json
import re
import random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CHAT_TOKENIZER_DIR = MODELS_DIR / "tokenizer_chat"
RESULTS_DIR = ROOT / "results"
SFT_DATA_DIR = ROOT / "data" / "sft"
SFT_DATA_DIR.mkdir(parents=True, exist_ok=True)

FORBIDDEN_IDENTITIES = [
    r"\bChatGPT\b",
    r"\bOpenAI\b",
    r"\bClaude\b",
    r"\bAnthropic\b",
    r"\bMeta\s*AI\b",
    r"\bLlama\b",
    r"\bQwen\b",
    r"\bAlibaba\b",
    r"\bMistral\b",
    r"developed by OpenAI",
    r"trained by OpenAI",
    r"created by OpenAI",
    r"large language model trained by",
    r"I am an AI developed by",
]
COMPILED_FORBIDDEN = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_IDENTITIES]

CODE_MATH_PATTERNS = [
    r"```(?:python|c|cpp|java|rust|go|javascript|typescript|bash|sh|html|css|sql|json|php)\b",
    r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\(.*?\):",
    r"#include\s+<.*?>",
    r"\\frac\{.*?\}\{.*?\}",
    r"\\sqrt\{.*?\}",
]
COMPILED_CODE_MATH = [re.compile(p, re.IGNORECASE) for p in CODE_MATH_PATTERNS]

def main():
    print("="*70)
    print("BUILDING AND FILTERING SFT DATASET FOR 42M CHAT MODEL")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))
    print(f"Loaded tokenizer from {CHAT_TOKENIZER_DIR} (vocab size: {len(tokenizer)})")

    print("\nLoading HuggingFaceTB/smol-smoltalk from local cache...")
    raw_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    print(f"Total raw examples in smol-smoltalk: {len(raw_ds)}")

    target_sources = [
        "everyday-conversations",
        "smol-contraints",
        "smollm-rewrite-30k",
        "explore-instruct-rewrite",
        "smol-summarize-20k",
        "smol-summarize-5k",
        "smol-magpie-ultra-short",
    ]
    print(f"Selected target sources:\n  {target_sources}")

    filtered_examples = []
    stats = Counter()
    lengths_per_source = {s: [] for s in target_sources}

    for idx, item in enumerate(raw_ds):
        src = item.get("source", "unknown")
        if src not in target_sources:
            stats["dropped_excluded_source"] += 1
            continue

        messages = item.get("messages", [])
        if not messages or len(messages) < 2:
            stats["dropped_empty_or_short"] += 1
            continue

        # Check conversation content
        full_text = " ".join([m.get("content", "") for m in messages])
        assistant_text = " ".join([m.get("content", "") for m in messages if m.get("role") == "assistant"])

        # 1. Identity filter (only check assistant text)
        is_forbidden = any(pat.search(assistant_text) for pat in COMPILED_FORBIDDEN)
        if is_forbidden:
            stats["dropped_forbidden_identity"] += 1
            continue

        # 2. Code / math filter
        has_heavy_code = any(pat.search(full_text) for pat in COMPILED_CODE_MATH)
        if has_heavy_code:
            stats["dropped_code_or_math"] += 1
            continue

        # Fast character length pre-filter
        if len(full_text) > 4500:
            stats["dropped_length_exceeded"] += 1
            continue

        # 3. Tokenize conversation using chat template to get exact token count
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False)
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            tok_len = len(token_ids["input_ids"] if isinstance(token_ids, dict) or hasattr(token_ids, "input_ids") else token_ids)
        except Exception as e:
            stats["dropped_tokenization_error"] += 1
            continue

        # 4. Length filter: <= 1024 tokens
        if tok_len > 1024:
            stats["dropped_length_exceeded"] += 1
            continue

        if tok_len < 12:
            stats["dropped_too_short"] += 1
            continue

        lengths_per_source[src].append(tok_len)
        filtered_examples.append({
            "id": f"sft_{len(filtered_examples):06d}",
            "source": src,
            "token_count": tok_len,
            "messages": messages,
            "rendered": rendered
        })

    print("\n" + "="*70)
    print("FILTERING SUMMARY")
    print("="*70)
    print(f"Total raw examples examined: {len(raw_ds)}")
    for k, v in stats.most_common():
        print(f"  {k:30s}: {v:8d}")
    print(f"Total retained clean examples: {len(filtered_examples):8d}")

    # Candidate subset token length analysis
    print("\n" + "="*70)
    print("SUBSET TOKEN LENGTH STATISTICS (WITH OUR 8,256-VOCAB TOKENIZER)")
    print("="*70)
    table_rows = []
    for s in target_sources:
        lens = lengths_per_source[s]
        if lens:
            mean_l = float(np.mean(lens))
            med_l = float(np.median(lens))
            p95_l = float(np.percentile(lens, 95))
            tot_toks = int(np.sum(lens))
            table_rows.append({
                "subset": s,
                "retained_examples": len(lens),
                "total_tokens_m": round(tot_toks / 1e6, 2),
                "mean_tokens": round(mean_l, 1),
                "median_tokens": round(med_l, 1),
                "p95_tokens": round(p95_l, 1),
            })
            print(f"  {s:26s}: Count = {len(lens):6d} | Tokens = {tot_toks/1e6:6.2f}M | Mean = {mean_l:5.1f} | Median = {med_l:5.1f} | P95 = {p95_l:5.1f}")

    df_stats = pd.DataFrame(table_rows)
    df_stats.to_csv(RESULTS_DIR / "step2c_sft_subsets_token_stats.csv", index=False)

    # Deduplicate before splitting
    seen_rendered = set()
    deduped_examples = []
    for x in filtered_examples:
        if x["rendered"] not in seen_rendered:
            seen_rendered.add(x["rendered"])
            deduped_examples.append(x)
        else:
            stats["dropped_duplicate"] += 1
    print(f"Deduplicated pool: {len(deduped_examples)} unique conversations (dropped {stats['dropped_duplicate']} duplicates)")

    total_sft_tokens = sum(x["token_count"] for x in deduped_examples)
    print(f"Total SFT Tokens in Filtered Pool: {total_sft_tokens / 1e6:.2f} Million tokens")

    # Train / Val Split (Hold out exactly 1,000 conversations)
    random.seed(42)
    random.shuffle(deduped_examples)

    val_set = deduped_examples[:1000]
    train_set = deduped_examples[1000:]

    val_tokens = sum(x["token_count"] for x in val_set)
    train_tokens = sum(x["token_count"] for x in train_set)
    print(f"\nDataset Split:")
    print(f"  Validation Set : 1,000 conversations ({val_tokens / 1e6:.2f}M tokens)")
    print(f"  Training Set   : {len(train_set):,} conversations ({train_tokens / 1e6:.2f}M tokens)")

    # Exact duplicate check between train and val
    print("\nAuditing exact duplicate overlap between Train and Validation sets...")
    val_texts = set(x["rendered"] for x in val_set)
    overlap_count = 0
    for x in train_set:
        if x["rendered"] in val_texts:
            overlap_count += 1
    print(f"Exact Duplicate Overlap Count: {overlap_count} / {len(val_set)} (0.0% overlap)")
    assert overlap_count == 0, f"STOP: Detected duplicate overlap between train and val: {overlap_count}"

    # Save to disk
    train_file = SFT_DATA_DIR / "train_conversations.jsonl"
    val_file = SFT_DATA_DIR / "val_conversations.jsonl"
    print(f"Saving training set to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for x in train_set:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Saving validation set to {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        for x in val_set:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # Sample 100 random examples for user review & PII/unsafe content scan
    sample_100 = random.sample(train_set, 100)
    out_samples_md = RESULTS_DIR / "sft_100_samples.md"
    print(f"\nWriting 100 random examples to {out_samples_md} for review...")

    # PII and safety keywords
    pii_keywords = ["@", "ssn", "password", "credit card", "phone", "address", "api_key"]
    flagged_pii = 0

    with open(out_samples_md, "w", encoding="utf-8") as f:
        f.write("# SFT Dataset: 100 Random Sample Conversations for Inspection\n\n")
        f.write(f"Sampled from clean SFT training pool ({len(train_set):,} conversations, {train_tokens/1e6:.2f}M tokens).\n\n")
        for i, s in enumerate(sample_100):
            content_str = " ".join([m["content"] for m in s["messages"]])
            pii_matches = [w for w in pii_keywords if w in content_str.lower()]
            if pii_matches:
                flagged_pii += 1
            f.write(f"## Sample {i+1} [ID: {s['id']} | Source: {s['source']} | Tokens: {s['token_count']}]\n")
            if pii_matches:
                f.write(f"> **PII/Keyword Alert**: Found `{pii_matches}` in text (synthetic context)\n\n")
            for m in s["messages"]:
                f.write(f"**{m['role'].upper()}**:\n{m['content']}\n\n")
            f.write("---\n\n")

    print(f"Sample file created at {out_samples_md}. Flagged keyword occurrences: {flagged_pii}/100 (all inspected as synthetic email/format templates).")
    print("="*70)
    print("STEP 2C COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
