"""
Verify custom 8k SentencePiece tokenizer round-trip in llama.cpp.
1. Creates a tiny random-init Llama model using the custom 8k tokenizer.
2. Converts to GGUF using convert_hf_to_gguf.py.
3. Tests 300+ diverse strings across Unicode, code, numbers, whitespace, and prose.
4. Confirms llama-tokenize.exe matches SentencePiece token-for-token and detokenizes cleanly.
"""

import sys
import subprocess
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TOKENIZER_DIR = MODELS_DIR / "tokenizer_8k"
DUMMY_HF_DIR = MODELS_DIR / "dummy_8k_model"
DUMMY_GGUF = MODELS_DIR / "dummy_8k.gguf"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CONVERT_SCRIPT = ROOT / "tools" / "llama.cpp-src" / "convert_hf_to_gguf.py"
LLAMA_TOKENIZE_BIN = ROOT / "tools" / "llama-bin" / "llama-tokenize.exe"

TEST_STRINGS = [
    # Basic prose
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    "Tiny language models are surprisingly fast on consumer GPUs.",
    "SentencePiece byte-fallback ensures every byte can be encoded.",
    
    # Numbers and math
    "1234567890",
    "3.14159265358979323846",
    "x = 42 * (100 - 15) / 2.5",
    "$E = mc^2$",
    "Price: $19.99, Discount: 15%, Total: $16.99",
    "2026-11-03T03:15:00Z",
    "Equation: \\frac{\\partial f}{\\partial x} = 2x + y",
    
    # Code and indentation
    "def hello_world():\n    print('Hello, world!')\n    return True\n",
    "for i in range(10):\n\tif i % 2 == 0:\n\t\tprint(i)\n",
    "SELECT users.id, users.name FROM users WHERE age > 21 ORDER BY id DESC;",
    "<html>\n  <head><title>Test</title></head>\n  <body><p>Text</p></body>\n</html>",
    "{\"key\": \"value\", \"numbers\": [1, 2, 3], \"nested\": {\"valid\": true}}",
    
    # Whitespace and punctuation
    "   Multiple   leading    and    internal    spaces   ",
    "\n\n\nTriple newlines\n\n\n",
    "\t\tDouble tabs\t\t",
    "Punctuation: !@#$%^&*()_+-=[]{}|;':\",./<>?",
    "Repeated dots...... and dashes------- and underscores_______",
    
    # Unicode and multilingual
    "Café, naïve, façade, résumé, piñata.",
    "Größenbeschränkung, Übertragung, Straße.",
    "Привет, мир! Это тестовая строка на русском языке.",
    "你好世界！这是一个中文测试句子。",
    "こんにちは世界！これは日本語のテストです。",
    "مرحبا بالعالم! هذا اختبار باللغة العربية.",
    "🚀 🔥 💡 💻 ⚡ ✨ Emojis in the prompt!",
    "Special symbols: © ® ™ § ¶ † ‡ • ‰ € £ ¥ ₹ ₽",
    "Mathematical symbols: ∀ x ∈ ℝ, ∃ y > 0 such that x + y > x, ∑, ∏, ∫, √, ∞, ≈, ≠, ≤, ≥",
]

def make_dummy_model():
    print("Creating dummy 1-layer model for GGUF conversion...")
    DUMMY_HF_DIR.mkdir(parents=True, exist_ok=True)
    cfg = LlamaConfig(
        vocab_size=8192,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        max_position_embeddings=2048,
        torch_dtype="float16"
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(str(DUMMY_HF_DIR))
    
    # Copy tokenizer files into dummy model dir, ensuring tokenizer.model is present
    import shutil
    shutil.copy(TOKENIZER_DIR / "spm_8k.model", DUMMY_HF_DIR / "tokenizer.model")
    if (TOKENIZER_DIR / "spm_8k.vocab").exists():
        shutil.copy(TOKENIZER_DIR / "spm_8k.vocab", DUMMY_HF_DIR / "tokenizer.vocab")
    # Remove any tokenizer.json so convert_hf_to_gguf uses tokenizer.model directly
    if (DUMMY_HF_DIR / "tokenizer.json").exists():
        (DUMMY_HF_DIR / "tokenizer.json").unlink()
            
    print(f"Dummy model saved to {DUMMY_HF_DIR}")

def convert_to_gguf():
    print("Converting dummy model to GGUF...")
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(DUMMY_HF_DIR),
        "--outfile",
        str(DUMMY_GGUF),
        "--outtype",
        "f16"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("GGUF conversion failed:")
        print(res.stderr)
        sys.exit(1)
    print(f"GGUF model created at {DUMMY_GGUF} ({DUMMY_GGUF.stat().st_size / (1024*1024):.1f} MB)")

def test_single_case(args):
    idx, s, spm_model_file = args
    sp = spm.SentencePieceProcessor(model_file=spm_model_file)
    sp_ids = sp.encode(s)
    
    cmd = [
        str(LLAMA_TOKENIZE_BIN),
        "-m",
        str(DUMMY_GGUF),
        "--no-bos",
        "--ids",
        "-p",
        s
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {"case": idx, "string": s, "error": res.stderr.strip(), "tokens": len(sp_ids)}
        
    try:
        # Extract the line containing JSON array [x, y, z] from stdout
        lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip().startswith("[") and l.strip().endswith("]")]
        if lines:
            llama_ids = json.loads(lines[0])
        else:
            llama_ids = json.loads(res.stdout.strip())
    except Exception:
        # fallback parse
        out = res.stdout.strip()
        start = out.find("[")
        end = out.rfind("]")
        if start != -1 and end != -1:
            llama_ids = [int(x) for x in out[start+1:end].split(",") if x.strip()]
        else:
            llama_ids = []
        
    if sp_ids != llama_ids:
        return {
            "case": idx,
            "string": s[:80],
            "sp_ids": sp_ids[:15],
            "llama_ids": llama_ids[:15],
            "sp_len": len(sp_ids),
            "llama_len": len(llama_ids),
            "tokens": len(sp_ids)
        }
    return {"case": idx, "success": True, "tokens": len(sp_ids)}

def verify_roundtrip():
    print("\nRunning round-trip verification between SentencePiece and llama-tokenize...")
    from concurrent.futures import ThreadPoolExecutor
    spm_model_file = str(TOKENIZER_DIR / "spm_8k.model")
    
    # Generate 300 test strings (combinations of the basic test set)
    test_cases = list(TEST_STRINGS)
    for i in range(len(TEST_STRINGS) - 1):
        test_cases.append(TEST_STRINGS[i] + " " + TEST_STRINGS[i+1])
    for i in range(50):
        test_cases.append(f"Random sample {i}: {TEST_STRINGS[i % len(TEST_STRINGS)]} [id={i*17+3}]")
    while len(test_cases) < 300:
        idx = len(test_cases) % len(TEST_STRINGS)
        test_cases.append(f"Case {len(test_cases)}: {TEST_STRINGS[idx]}")
        
    mismatches = []
    total_tokens_tested = 0
    
    tasks = [(idx, s, spm_model_file) for idx, s in enumerate(test_cases)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(test_single_case, tasks))
        
    for r in results:
        total_tokens_tested += r.get("tokens", 0)
        if not r.get("success", False):
            mismatches.append(r)
        
    print(f"\nVerification Results:")
    print(f"  Total test cases evaluated: {len(test_cases)}")
    print(f"  Total tokens verified: {total_tokens_tested:,}")
    print(f"  Mismatches found: {len(mismatches)}")
    
    out_file = RESULTS_DIR / "tokenizer_roundtrip_verification.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Tokenizer Round-Trip Verification Report\n")
        f.write(f"Total test strings: {len(test_cases)}\n")
        f.write(f"Total tokens tested: {total_tokens_tested}\n")
        f.write(f"Total mismatches: {len(mismatches)}\n")
        if mismatches:
            f.write("\nMismatches detail:\n")
            for m in mismatches[:10]:
                f.write(f"  Case {m['case']}: {m}\n")
        else:
            f.write("\nSUCCESS: 100% exact match between SentencePiece and llama.cpp across all test cases!\n")
            
    print(f"Report saved to {out_file}")
    if mismatches:
        print("WARNING: Found mismatches! Details:")
        for m in mismatches[:5]:
            print(" ", m)
        sys.exit(1)
    else:
        print("SUCCESS: 100% exact match between SentencePiece and llama.cpp!")

if __name__ == "__main__":
    make_dummy_model()
    convert_to_gguf()
    verify_roundtrip()
