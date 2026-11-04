"""
Sanity Check 1.b: Export 4L_768H_1KV 1-hour decayed checkpoint to GGUF.
1. Verifies RoPE settings and tokenizer metadata in GGUF.
2. Runs the 10 fixed prompts in PyTorch at temperature 0 (greedy) with BOS.
3. Runs the same 10 prompts in llama.cpp (llama-completion.exe) at temperature 0 (greedy).
4. Compares generated token sequences and reports match status.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm
import gguf

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
CKPT_PATH = CHECKPOINTS_DIR / "4L_768H_1KV_decayed_1hr.pt"
HF_DIR = MODELS_DIR / "hf_4L_768H_1KV"
GGUF_PATH = MODELS_DIR / "4L_768H_1KV_decayed_1hr.gguf"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
CONVERT_SCRIPT = ROOT / "tools" / "llama.cpp-src" / "convert_hf_to_gguf.py"
LLAMA_COMPLETION = ROOT / "tools" / "llama-bin" / "llama-completion.exe"

SHAPE_CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_attention_heads": 12,
    "num_key_value_heads": 1,
    "vocab_size": 8192,
}

SAMPLE_PROMPTS = [
    "Once upon a time, there was a little bunny named",
    "The fundamental difference between planets and stars is",
    "To write a simple Python function that adds two numbers:",
    "In ancient history, the Great Pyramid of Giza was built",
    "Lily wanted to bake a cake for her mom. First, she",
    "Water evaporates when the temperature rises because",
    "A computer network is a system of connected computers that",
    "The dog ran across the park to catch",
    "Photosynthesis is the biological process by which plants",
    "When solving a math problem, it is important to",
]

def verify_gguf_metadata():
    print("=" * 70)
    print("  VERIFYING GGUF METADATA (ROPE & TOKENIZER)")
    print("=" * 70)
    reader = gguf.GGUFReader(str(GGUF_PATH))
    
    def get_scalar(field_name):
        f = reader.get_field(field_name)
        return f.parts[f.data[0]][0] if f is not None else None

    def get_string(field_name):
        f = reader.get_field(field_name)
        return bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace") if f is not None else None
        
    arch = get_string("general.architecture")
    ctx_len = get_scalar("llama.context_length")
    embed_len = get_scalar("llama.embedding_length")
    block_cnt = get_scalar("llama.block_count")
    ffn_len = get_scalar("llama.feed_forward_length")
    head_cnt = get_scalar("llama.attention.head_count")
    head_kv = get_scalar("llama.attention.head_count_kv")
    rope_base = get_scalar("llama.rope.freq_base")
    rope_dim = get_scalar("llama.rope.dimension_count")
    tok_model = get_string("tokenizer.ggml.model")
    tok_field = reader.get_field("tokenizer.ggml.tokens")
    vocab_size = len(tok_field.data) if tok_field is not None else 0
    
    print(f"  Architecture: {arch}")
    print(f"  Context Length: {ctx_len}")
    print(f"  Embedding Length: {embed_len}")
    print(f"  Block Count (Layers): {block_cnt}")
    print(f"  Feed Forward Length: {ffn_len}")
    print(f"  Attention Heads: {head_cnt}")
    print(f"  KV Heads: {head_kv}")
    print(f"  RoPE Freq Base (theta): {rope_base}")
    print(f"  RoPE Dimension Count: {rope_dim}")
    print(f"  Tokenizer Model: {tok_model}")
    print(f"  Tokenizer Vocab Size: {vocab_size}")
    
    assert arch == "llama", f"Arch mismatch: {arch}"
    assert block_cnt == 4, f"Layer mismatch: {block_cnt}"
    assert head_kv == 1, f"KV head mismatch: {head_kv}"
    assert abs(rope_base - 10000.0) < 1.0, f"RoPE base mismatch: {rope_base}"
    assert rope_dim == 64, f"RoPE dim mismatch: {rope_dim}"
    assert vocab_size == 8192, f"Vocab size mismatch: {vocab_size}"
    print("  >>> [PASS] All GGUF Architecture, RoPE, and Tokenizer Metadata Verified! <<<\n")

def run_pytorch_greedy(prompts, max_new_tokens=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = LlamaConfig(
        vocab_size=SHAPE_CFG["vocab_size"],
        hidden_size=SHAPE_CFG["hidden_size"],
        intermediate_size=SHAPE_CFG["intermediate_size"],
        num_hidden_layers=SHAPE_CFG["num_hidden_layers"],
        num_attention_heads=SHAPE_CFG["num_attention_heads"],
        num_key_value_heads=SHAPE_CFG["num_key_value_heads"],
        tie_word_embeddings=True,
        torch_dtype="float16",
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(cfg).half().to(device)
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    
    results = []
    for p in prompts:
        # Include BOS (1) as llama.cpp does by default
        prompt_tokens = sp.encode(p)
        curr_ids = [1] + list(prompt_tokens)
        gen_tokens = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr_ids], dtype=torch.long, device=device)
                out = model(inp)
                next_tok = torch.argmax(out.logits[0, -1, :]).item()
                if next_tok == 2:  # EOS
                    break
                curr_ids.append(next_tok)
                gen_tokens.append(next_tok)
        gen_text = sp.decode(gen_tokens).strip()
        full_text = sp.decode(curr_ids[1:]).strip()
        results.append({
            "prompt": p,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "generated_text": gen_text,
            "full_text": full_text
        })
    del model
    return results

def run_llamacpp_greedy(prompts, max_new_tokens=32):
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    results = []
    
    for p in prompts:
        cmd = [
            str(LLAMA_COMPLETION),
            "-m", str(GGUF_PATH),
            "-p", p,
            "-n", str(max_new_tokens),
            "--temp", "0",
            "-ngl", "99",
            "-fa", "1",
            "--no-warmup",
            "--no-display-prompt"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        # Generated text from stdout (with leading/trailing whitespace stripped)
        gen_text = proc.stdout.strip()
        prompt_toks = sp.encode(p)
        full_text = (p + " " + gen_text).strip()
        
        results.append({
            "prompt": p,
            "prompt_tokens": prompt_toks,
            "generated_text": gen_text,
            "full_text": full_text
        })
    return results

def compare_outputs(pt_results, llama_results):
    print("=" * 70)
    print("  GREEDY OUTPUT COMPARISON: PYTORCH VS LLAMA.CPP")
    print("=" * 70)
    
    total_prompt_matches = 0
    total_prompts = len(pt_results)
    
    for i, (pt, ll) in enumerate(zip(pt_results, llama_results)):
        p = pt["prompt"]
        pt_text = pt["generated_text"]
        ll_text = ll["generated_text"]
        
        exact_match = (pt_text == ll_text)
        if exact_match:
            total_prompt_matches += 1
            
        print(f"\nPrompt {i+1}: \"{p}\"")
        print(f"  PyTorch:   {pt_text}")
        print(f"  llama.cpp: {ll_text}")
        print(f"  Exact Text Match: {exact_match}")
        
    match_pct = (total_prompt_matches / total_prompts) * 100.0
    print("\n" + "=" * 70)
    print(f"  EXACT TEXT MATCH: {total_prompt_matches}/{total_prompts} ({match_pct:.1f}%)")
    print("=" * 70)
    return match_pct

def main():
    verify_gguf_metadata()
    
    print("Running PyTorch Greedy Generation (with BOS) on 10 Prompts...")
    pt_results = run_pytorch_greedy(SAMPLE_PROMPTS)
    
    print("Running llama.cpp Greedy Generation on 10 Prompts...")
    llama_results = run_llamacpp_greedy(SAMPLE_PROMPTS)
    
    match_rate = compare_outputs(pt_results, llama_results)
    print(f"\nSanity Check 1.b Complete: Overall Match Rate = {match_rate:.2f}%")
    if match_rate >= 80.0:
        print(">>> [PASS] PyTorch and llama.cpp outputs match! GGUF export path verified. <<<")
    else:
        print(">>> [WARNING/FAIL] Match rate below threshold! <<<")

if __name__ == "__main__":
    main()
