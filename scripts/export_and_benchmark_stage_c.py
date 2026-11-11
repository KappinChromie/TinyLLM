"""
Step 7: Stage C GGUF Export, Parity Verification, 32k Context Test, and llama-bench Benchmarking
- Exports models/checkpoints/stageC/stageC_decayed_final.pt to Hugging Face format and GGUF (F16, Q8_0, Q4_0).
- Runs 10-prompt parity test against PyTorch.
- Tests 32k context in llama.cpp on a long text passage.
- Benchmarks decode speed with llama-bench.exe at depths 0, 8k, 16k, 32k (128 generated tokens, 5 reps).
- Outputs results/stageC_decode_speed.csv and results/stageC_gguf_parity.txt.
"""

import os
import sys
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
STAGE_C_CKPT = MODELS_DIR / "checkpoints" / "stageC" / "stageC_decayed_final.pt"
TOKENIZER_DIR = MODELS_DIR / "tokenizer_8k"
RESULTS_DIR = ROOT / "results"
TOOLS_DIR = ROOT / "tools" / "llama-bin"

LLAMA_COMPLETION = TOOLS_DIR / "llama-completion.exe"
LLAMA_BENCH = TOOLS_DIR / "llama-bench.exe"
LLAMA_QUANTIZE = TOOLS_DIR / "llama-quantize.exe"
LLAMA_PERPLEXITY = TOOLS_DIR / "llama-perplexity.exe"
CONVERT_SCRIPT = ROOT / "tools" / "llama.cpp-src" / "convert_hf_to_gguf.py"

OUT_PARITY_TXT = RESULTS_DIR / "stageC_gguf_parity.txt"
OUT_SPEED_CSV = RESULTS_DIR / "stageC_decode_speed.csv"
OUT_QUANT_QUALITY_CSV = RESULTS_DIR / "stageC_quantization_quality.csv"
OUT_METADATA_TXT = RESULTS_DIR / "stageC_gguf_metadata.txt"

FIXED_PROMPTS_10 = [
    "Once upon a time, there was a little bunny named ",
    "The fundamental difference between planets and stars is ",
    "To write a simple Python function that adds two numbers: ",
    "In ancient history, the Great Pyramid of Giza was built ",
    "Lily wanted to bake a cake for her mom. First, she ",
    "Water evaporates when the temperature rises because ",
    "A computer network is a system of connected computers that ",
    "The dog ran across the park to catch ",
    "Photosynthesis is the biological process by which plants ",
    "When solving a math problem, it is important to ",
]

def export_to_hf_and_gguf(ckpt_path, hf_dir, gguf_dir, rope_theta=1000000.0):
    print(f"\n--- Exporting Stage C Decayed Checkpoint to HF & GGUF (RoPE Theta: {rope_theta:,.0f}) ---", flush=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    
    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=32768,
        rope_theta=rope_theta,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    model.save_pretrained(hf_dir)
    
    # Copy tokenizer files
    shutil.copyfile(TOKENIZER_DIR / "tokenizer.model", hf_dir / "tokenizer.model")
    if (TOKENIZER_DIR / "tokenizer_config.json").exists():
        shutil.copyfile(TOKENIZER_DIR / "tokenizer_config.json", hf_dir / "tokenizer_config.json")
    if (hf_dir / "tokenizer.json").exists():
        (hf_dir / "tokenizer.json").unlink()
        
    f16_gguf = gguf_dir / "stageC_decayed_f16.gguf"
    q8_gguf = gguf_dir / "stageC_decayed_q8_0.gguf"
    q4_gguf = gguf_dir / "stageC_decayed_q4_0.gguf"
    
    if f16_gguf.exists() and q8_gguf.exists() and q4_gguf.exists():
        print(f"Reusing existing GGUF files: {f16_gguf.name}, {q8_gguf.name}, {q4_gguf.name}...")
        return f16_gguf, q8_gguf, q4_gguf
    
    print(f"Converting HF model to GGUF F16 at {f16_gguf}...", flush=True)
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(hf_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "tools" / "llama.cpp-src" / "gguf-py") + (";" if os.name == "nt" else ":") + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"GGUF convert failed: {res.stderr}")
    print(f"  F16 GGUF created: {f16_gguf} ({f16_gguf.stat().st_size / 1e6:.1f} MB)", flush=True)
    
    # Quantize to Q8_0
    print(f"Quantizing to Q8_0 at {q8_gguf}...", flush=True)
    cmd_q8 = [str(LLAMA_QUANTIZE), str(f16_gguf.relative_to(ROOT)), str(q8_gguf.relative_to(ROOT)), "q8_0"]
    subprocess.run(cmd_q8, check=True)
    print(f"  Q8_0 GGUF created: {q8_gguf} ({q8_gguf.stat().st_size / 1e6:.1f} MB)", flush=True)
    
    # Quantize to Q4_0
    print(f"Quantizing to Q4_0 at {q4_gguf}...", flush=True)
    cmd_q4 = [str(LLAMA_QUANTIZE), str(f16_gguf.relative_to(ROOT)), str(q4_gguf.relative_to(ROOT)), "q4_0"]
    subprocess.run(cmd_q4, check=True)
    print(f"  Q4_0 GGUF created: {q4_gguf} ({q4_gguf.stat().st_size / 1e6:.1f} MB)", flush=True)
    
    return f16_gguf, q8_gguf, q4_gguf

def generate_pytorch(model, sp, prompt, device, max_new_tokens=32):
    model.eval()
    input_ids = sp.encode_as_ids(prompt)
    curr_ids = list(input_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([curr_ids[-32768:]], dtype=torch.long, device=device)
            out = model(inp)
            next_tok = torch.argmax(out.logits[0, -1, :]).item()
            curr_ids.append(next_tok)
            if next_tok == sp.eos_id():
                break
    new_ids = curr_ids[len(input_ids):]
    return new_ids, sp.decode(new_ids)

def generate_llamacpp(gguf_path, prompt, max_new_tokens=32):
    rel_m = str(Path(gguf_path).relative_to(ROOT))
    cmd = [
        str(LLAMA_COMPLETION),
        "-m", rel_m,
        "-p", prompt,
        "-n", str(max_new_tokens),
        "-ngl", "99",
        "-c", "32768",
        "--temp", "0.0",
        "--top-k", "0",
        "--top-p", "1.0",
        "--min-p", "0.0",
        "--repeat-penalty", "1.0",
        "--override-kv", "tokenizer.ggml.add_bos_token=bool:false",
        "--no-warmup"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    raw_out = res.stdout
    prompt_pos = raw_out.find(prompt)
    if prompt_pos != -1:
        gen = raw_out[prompt_pos + len(prompt):]
    else:
        gen = raw_out
    for marker in ["[end of text]", "\nllama_perf_context_print", "\nllama_perf_sampler_print"]:
        if marker in gen:
            gen = gen[:gen.find(marker)]
    return gen.strip()

def run_parity_check(model, sp, f16_gguf, device):
    print("\n--- Running 10-Prompt Parity Check: PyTorch float16 vs llama.cpp (F16) ---", flush=True)
    report_lines = []
    report_lines.append("=== Stage C GGUF vs PyTorch Parity Verification Report ===")
    matches = 0
    
    for idx, prompt in enumerate(FIXED_PROMPTS_10, 1):
        pt_ids, pt_text = generate_pytorch(model, sp, prompt, device, max_new_tokens=32)
        llama_text = generate_llamacpp(f16_gguf, prompt, max_new_tokens=32)
        
        is_match = (pt_text.strip() == llama_text.strip()) or (pt_text.strip().startswith(llama_text.strip())) or (llama_text.strip().startswith(pt_text.strip()))
        if is_match:
            matches += 1
            status = "MATCH"
        else:
            status = "MISMATCH"
            
        print(f"[{status}] Prompt {idx:2d}: '{prompt[:35]}...'")
        print(f"  PyTorch : {pt_text.strip()[:60]}")
        print(f"  llama.cpp: {llama_text.strip()[:60]}")
        
        report_lines.append(f"\n--- Prompt {idx} [{status}] ---")
        report_lines.append(f"Prompt: {prompt}")
        report_lines.append(f"PyTorch Output : {pt_text.strip()}")
        report_lines.append(f"llama.cpp Output: {llama_text.strip()}")
        
    parity_rate = (matches / len(FIXED_PROMPTS_10)) * 100.0
    summary = f"\nFinal Parity Result: {matches}/{len(FIXED_PROMPTS_10)} ({parity_rate:.1f}%) matches."
    print(summary)
    report_lines.append(summary)
    
    with open(OUT_PARITY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Parity report written to {OUT_PARITY_TXT}")

def verify_gguf_metadata(f16_path, q8_path, q4_path):
    print("\n--- Verifying GGUF Metadata ---", flush=True)
    sys.path.insert(0, str(ROOT / "tools" / "llama.cpp-src" / "gguf-py"))
    import gguf
    
    meta_records = []
    for label, path in [("F16", f16_path), ("Q8_0", q8_path), ("Q4_0", q4_path)]:
        reader = gguf.GGUFReader(str(path))
        ctx_len = None
        rope_freq = None
        blocks = None
        heads = None
        heads_kv = None
        embd = None
        for field in reader.fields.values():
            if field.name == "llama.context_length":
                ctx_len = field.parts[-1][0]
            elif field.name == "llama.rope.freq_base":
                rope_freq = field.parts[-1][0]
            elif field.name == "llama.block_count":
                blocks = field.parts[-1][0]
            elif field.name == "llama.attention.head_count":
                heads = field.parts[-1][0]
            elif field.name == "llama.attention.head_count_kv":
                heads_kv = field.parts[-1][0]
            elif field.name == "llama.embedding_length":
                embd = field.parts[-1][0]
                
        print(f"[{label}] File: {path.name}")
        print(f"  Context Length: {ctx_len}")
        print(f"  RoPE Freq Base: {rope_freq}")
        print(f"  Layers (blocks): {blocks}")
        print(f"  Attention Heads: {heads} (KV Heads: {heads_kv})")
        print(f"  Embedding Dim: {embd}")
        
        assert ctx_len == 32768, f"Context length expected 32768, got {ctx_len}"
        assert blocks == 6, f"Block count expected 6, got {blocks}"
        assert heads == 12, f"Heads expected 12, got {heads}"
        assert heads_kv == 1, f"KV Heads expected 1, got {heads_kv}"
        assert embd == 768, f"Embedding expected 768, got {embd}"
        
        meta_records.append({
            "quant": label,
            "filename": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "context_length": ctx_len,
            "rope_freq_base": float(rope_freq),
            "layers": blocks,
            "heads": heads,
            "heads_kv": heads_kv,
            "embedding_dim": embd,
        })
    df_meta = pd.DataFrame(meta_records)
    with open(OUT_METADATA_TXT, "w", encoding="utf-8") as f:
        f.write(df_meta.to_string() + "\n")
    print(f"Metadata verification passed for all 3 formats! Saved to {OUT_METADATA_TXT}\n")
    return df_meta

def test_32k_context_parity_and_loss(pt_model, f16_gguf, sp, device):
    print("\n--- 32k Context Perplexity & Loss Parity Test: PyTorch vs llama.cpp ---", flush=True)
    val_pg19 = np.memmap(ROOT / "data" / "shards" / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    raw_tokens = val_pg19[:65536].astype(np.int64)
    passage_text = sp.decode(list(raw_tokens))
    
    # 1. PyTorch 32k loss
    x = torch.from_numpy(raw_tokens[:32768]).unsqueeze(0).to(device)
    y = torch.from_numpy(raw_tokens[1:32769]).unsqueeze(0).to(device)
    pt_model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = pt_model(x)
            pt_loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1)).item()
    pt_ppl = math.exp(pt_loss)
    print(f"  PyTorch 32k Evaluation: Loss = {pt_loss:.4f} CE | PPL = {pt_ppl:.2f}")
    
    # 2. llama.cpp 32k perplexity
    temp_file = RESULTS_DIR / "temp_32k_passage.txt"
    with open(temp_file, "w", encoding="utf-8", errors="ignore") as f:
        f.write(passage_text)
        
    cmd = [
        str(LLAMA_PERPLEXITY),
        "-m", str(Path(f16_gguf).relative_to(ROOT)),
        "-f", str(Path(temp_file).relative_to(ROOT)),
        "-c", "32768",
        "-ngl", "99",
        "--chunks", "1"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    full_output = res.stdout + "\n" + res.stderr
    llama_ppl = None
    for line in full_output.split("\n"):
        if "Final estimate: PPL =" in line:
            parts = line.split("Final estimate: PPL =")[1].strip().split()
            llama_ppl = float(parts[0])
            break
        elif "[1]" in line and "," in line:
            part = line.split("[1]")[1].split(",")[0].strip()
            try:
                llama_ppl = float(part)
            except Exception:
                pass
            
    if llama_ppl is not None:
        llama_loss = math.log(llama_ppl)
        delta_loss = abs(pt_loss - llama_loss)
        print(f"  llama.cpp 32k Evaluation: Loss = {llama_loss:.4f} CE | PPL = {llama_ppl:.2f}")
        print(f"  Parity Delta (PyTorch vs llama.cpp): {delta_loss:.4f} CE")
    else:
        print(f"  Warning: Could not parse llama-perplexity output: {res.stdout[-300:]}")
        
    if temp_file.exists():
        temp_file.unlink()

def run_quantization_quality(f16_path, q8_path, q4_path, sp):
    print("\n--- Evaluating Quantization Quality (F16 vs Q8_0 vs Q4_0) ---", flush=True)
    shards_dir = ROOT / "data" / "shards"
    eval_sets = {
        "val_wikipedia_neutral": shards_dir / "val_wikipedia_neutral.bin",
        "val_fineweb_edu": shards_dir / "val_fineweb_edu.bin",
        "val_cosmopedia": shards_dir / "val_cosmopedia.bin",
        "val_tinystories": shards_dir / "val_tinystories.bin",
    }
    
    # Extract fixed 50k tokens per validation set and 32k PG-19 passage
    temp_files = {}
    for name, path in eval_sets.items():
        shard = np.memmap(path, dtype=np.uint16, mode="r")
        text = sp.decode(list(shard[:51200].astype(int)))
        tf = RESULTS_DIR / f"temp_{name}.txt"
        with open(tf, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)
        temp_files[name] = (tf, 2048)
        
    pg19_shard = np.memmap(shards_dir / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    pg19_text = sp.decode(list(pg19_shard[:65536].astype(int)))
    pg19_tf = RESULTS_DIR / "temp_val_pg19_32k.txt"
    with open(pg19_tf, "w", encoding="utf-8", errors="ignore") as f:
        f.write(pg19_text)
    temp_files["val_pg19_32k"] = (pg19_tf, 32768)
    
    models = [("F16", f16_path), ("Q8_0", q8_path), ("Q4_0", q4_path)]
    records = []
    
    for q_label, m_path in models:
        for ds_name, (tf_path, ctx_len) in temp_files.items():
            print(f"Evaluating {q_label} on {ds_name} (ctx {ctx_len})...", flush=True)
            cmd = [
                str(LLAMA_PERPLEXITY),
                "-m", str(Path(m_path).relative_to(ROOT)),
                "-f", str(Path(tf_path).relative_to(ROOT)),
                "-c", str(ctx_len),
                "-ngl", "99"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            full_output = res.stdout + "\n" + res.stderr
            ppl_val = None
            err_val = None
            for line in full_output.split("\n"):
                if "Final estimate: PPL =" in line:
                    parts = line.split("Final estimate: PPL =")[1].strip().split()
                    ppl_val = float(parts[0])
                    if len(parts) >= 3 and parts[1] == "+/-":
                        err_val = float(parts[2])
                    break
                elif "[1]" in line and "," in line and ppl_val is None:
                    # In case only 1 chunk was run
                    chunks = [c for c in line.split(",") if "[" in c and "]" in c]
                    if chunks:
                        last_c = chunks[-1]
                        try:
                            val_str = last_c.split("]")[1].strip()
                            ppl_val = float(val_str)
                        except Exception:
                            pass
            ce_val = round(math.log(ppl_val), 4) if ppl_val is not None else None
            records.append({
                "quant": q_label,
                "dataset": ds_name,
                "context_len": ctx_len,
                "perplexity": ppl_val,
                "perplexity_err": err_val,
                "ce_loss": ce_val,
            })
            print(f"  [{q_label} | {ds_name}] PPL = {ppl_val} | CE = {ce_val}")
            
    # Cleanup temp files
    for tf_path, _ in temp_files.values():
        if tf_path.exists():
            tf_path.unlink()
            
    df_q = pd.DataFrame(records)
    # Add delta vs F16
    f16_map = df_q[df_q["quant"] == "F16"].set_index("dataset")["ce_loss"].to_dict()
    df_q["delta_ce_vs_f16"] = df_q.apply(
        lambda r: round(r["ce_loss"] - f16_map.get(r["dataset"], r["ce_loss"]), 4) if r["ce_loss"] is not None else None,
        axis=1
    )
    df_q.to_csv(OUT_QUANT_QUALITY_CSV, index=False)
    print(f"\nSaved Quantization Quality Results to {OUT_QUANT_QUALITY_CSV}!\n")
    print(df_q.to_string())
    return df_q

def run_llama_bench(f16_path, q8_path, q4_path):
    print("\n--- Running llama-bench Decode Speed at Context Depths 0k, 8k, 16k, 32k ---", flush=True)
    models = [
        ("F16", f16_path),
        ("Q8_0", q8_path),
        ("Q4_0", q4_path),
    ]
    
    records = []
    for q_label, m_path in models:
        print(f"Benchmarking {q_label} with llama-bench (-n 256 -fa 1 -d 0,8192,16384,32768 -r 5)...", flush=True)
        cmd = [
            str(LLAMA_BENCH),
            "-m", str(Path(m_path).relative_to(ROOT)),
            "-ngl", "99",
            "-fa", "1",
            "-n", "256",
            "-p", "0",
            "-d", "0,8192,16384,32768",
            "-r", "5",
            "-o", "json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(res.stdout)
            for item in data:
                depth = item.get("n_depth", 0)
                avg_ts = item.get("avg_ts", 0.0)
                stddev_ts = item.get("stddev_ts", 0.0)
                print(f"  {q_label} @ Depth {depth:5d}: {avg_ts:.2f} +/- {stddev_ts:.2f} tok/s")
                records.append({
                    "quant": q_label,
                    "context_depth": depth,
                    "gen_tokens": 256,
                    "decode_tok_s": round(avg_ts, 2),
                    "decode_err_tok_s": round(stddev_ts, 2),
                })
        except Exception as e:
            print(f"  Error parsing JSON: {e}\nRaw output:\n{res.stdout[:400]}")
            
    df = pd.DataFrame(records)
    df.to_csv(OUT_SPEED_CSV, index=False)
    print(f"\nSaved llama-bench results to {OUT_SPEED_CSV}!\n")
    print(df.to_string())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage C Export, Parity, and llama-bench Execution on {device} ===")
    
    if not STAGE_C_CKPT.exists():
        raise FileNotFoundError(f"Final Stage C checkpoint {STAGE_C_CKPT} does not exist!")
        
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_DIR / "tokenizer.model"))
    
    # Check winning theta from Stage C mini-trial if available
    rope_csv = RESULTS_DIR / "rope_minitrial_stageC.csv"
    rope_theta = 1000000.0
    if rope_csv.exists():
        df_r = pd.read_csv(rope_csv)
        if "rope_theta" in df_r.columns:
            rope_theta = float(df_r["rope_theta"].iloc[0])
            
    hf_dir = MODELS_DIR / "hf_stageC_final"
    gguf_dir = MODELS_DIR / "gguf"
    
    f16_gguf, q8_gguf, q4_gguf = export_to_hf_and_gguf(STAGE_C_CKPT, hf_dir, gguf_dir, rope_theta=rope_theta)
    
    # Verify metadata on all 3 exported files
    verify_gguf_metadata(f16_gguf, q8_gguf, q4_gguf)
    
    # Load model in PyTorch for parity check and 32k loss comparison
    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="float16",
        _attn_implementation="sdpa",
        max_position_embeddings=32768,
        rope_theta=rope_theta,
    )
    pt_model = LlamaForCausalLM(config).half().to(device)
    raw = torch.load(STAGE_C_CKPT, map_location=device, weights_only=False)
    pt_model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    
    run_parity_check(pt_model, sp, f16_gguf, device)
    test_32k_context_parity_and_loss(pt_model, f16_gguf, sp, device)
    del pt_model
    torch.cuda.empty_cache()
    
    run_quantization_quality(f16_gguf, q8_gguf, q4_gguf, sp)
    run_llama_bench(f16_gguf, q8_gguf, q4_gguf)
    
    print("\nStage C Export, Verification, and Speed Benchmarking Finished Successfully!")

if __name__ == "__main__":
    main()
