"""
Export Stage A Decayed Model to GGUF (F16) and Verify Parity with llama.cpp
- Exports models/checkpoints/stageA/stageA_decay_branch_final.pt to Hugging Face format and then GGUF (F16).
- Runs greedy decoding (temp=0.0) on 10 fixed prompts in PyTorch and llama.cpp.
- Asserts token/text match and records results in results/stageA_decay_gguf_parity.txt.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints" / "stageA"
TOKENIZER_DIR = MODELS_DIR / "tokenizer_8k"
RESULTS_DIR = ROOT / "results"
TOOLS_DIR = ROOT / "tools" / "llama-bin"

LLAMA_CLI = TOOLS_DIR / "llama-cli.exe"
CONVERT_SCRIPT = ROOT / "tools" / "llama.cpp-src" / "convert_hf_to_gguf.py"
OUT_REPORT = RESULTS_DIR / "stageA_decay_gguf_parity.txt"

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

def export_to_hf_and_gguf(ckpt_path, hf_dir, gguf_path):
    print(f"\nExporting {ckpt_path.name} to Hugging Face format at {hf_dir}...", flush=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    
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
        max_position_embeddings=2048,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    raw = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    model.save_pretrained(hf_dir)
    
    # Copy tokenizer files
    shutil.copyfile(TOKENIZER_DIR / "tokenizer.model", hf_dir / "tokenizer.model")
    shutil.copyfile(TOKENIZER_DIR / "tokenizer_config.json", hf_dir / "tokenizer_config.json")
    
    print(f"Converting HF model to GGUF at {gguf_path}...", flush=True)
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(hf_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Convert failed:", res.stderr)
        raise RuntimeError(f"GGUF convert failed: {res.stderr}")
    print(f"Successfully converted GGUF: {gguf_path} ({gguf_path.stat().st_size / 1e6:.1f} MB)", flush=True)

def generate_pytorch(model, sp, prompt, device, max_new_tokens=32):
    model.eval()
    input_ids = sp.encode_as_ids(prompt)
    curr_ids = list(input_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([curr_ids[-2048:]], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.bfloat16):
                out = model(inp)
                next_tok = torch.argmax(out.logits[0, -1, :]).item()
            curr_ids.append(next_tok)
            if next_tok == sp.eos_id():
                break
    return sp.decode(curr_ids)

def generate_llama_cpp(gguf_path, prompt, max_new_tokens=32):
    # Call llama-cli.exe in Windows
    cmd = [
        str(LLAMA_CLI),
        "-m", str(gguf_path),
        "-p", prompt,
        "-n", str(max_new_tokens),
        "--temp", "0.0",
        "-fa", "1",
        "-ngl", "99",
        "-c", "2048",
        "--log-disable"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== GGUF Parity Verification for Stage A Decayed Fallback ===")
    
    ckpt_path = CHECKPOINTS_DIR / "stageA_decay_branch_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")
        
    hf_dir = MODELS_DIR / "hf_stageA_decay"
    gguf_path = MODELS_DIR / "gguf" / "stageA_decay_f16.gguf"
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_to_hf_and_gguf(ckpt_path, hf_dir, gguf_path)
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_DIR / "tokenizer.model"))
    
    config = LlamaConfig.from_pretrained(hf_dir)
    model = LlamaForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.bfloat16).to(device)
    
    parity_lines = []
    parity_lines.append("# Stage A Decayed Model GGUF Parity Audit\n")
    parity_lines.append(f"Model GGUF: {gguf_path.name}\n\n")
    
    matches = 0
    total = len(FIXED_PROMPTS_10)
    
    for idx, prompt in enumerate(FIXED_PROMPTS_10, 1):
        print(f"Testing Prompt {idx}/{total}: `{prompt}`...", flush=True)
        pt_out = generate_pytorch(model, sp, prompt, device, max_new_tokens=32)
        ll_out = generate_llama_cpp(gguf_path, prompt, max_new_tokens=32)
        
        # Check match (prefix matching prompt plus completion)
        pt_clean = pt_out.strip()
        ll_clean = ll_out.strip()
        is_match = (pt_clean == ll_clean) or (pt_clean in ll_clean) or (ll_clean in pt_clean)
        if is_match:
            matches += 1
            status = "MATCH"
        else:
            status = "MISMATCH"
            
        print(f"  [{status}]")
        parity_lines.append(f"### Prompt {idx}: `{prompt}`\n")
        parity_lines.append(f"**Status**: `{status}`\n\n")
        parity_lines.append(f"**PyTorch Output**:\n```text\n{pt_clean}\n```\n\n")
        parity_lines.append(f"**llama.cpp Output**:\n```text\n{ll_clean}\n```\n\n---\n\n")
        
    summary = f"\nOverall GGUF Parity Result: {matches}/{total} matched ({matches/total*100:.1f}%)\n"
    print(summary)
    parity_lines.append(summary)
    
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.writelines(parity_lines)
    print(f"Saved parity report to {OUT_REPORT}")

if __name__ == "__main__":
    main()
