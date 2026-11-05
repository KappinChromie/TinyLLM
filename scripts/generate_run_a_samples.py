"""
Generate Evaluation Samples for Run_A from final checkpoint
"""

import sys
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CKPT_PATH = MODELS_DIR / "checkpoints" / "phase5b" / "Run_A_final.pt"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
SAMPLES_MD = RESULTS_DIR / "phase5b_sample_generations.md"

PROMPTS = [
    # 10 Standard Prompts
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
    # 5 Basic Grammar Prompts
    "She opened the door and saw ",
    "The dog ran after the ball because ",
    "He went to the store to buy ",
    "They were very happy when they ",
    "Every morning, the boy eats ",
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Tokenizer from {TOKENIZER_MODEL}...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))

    print(f"Loading Model from {CKPT_PATH}...")
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
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()

    print("Generating samples...")
    samples = []
    for p in PROMPTS:
        input_ids = sp.encode_as_ids(p)
        curr_ids = list(input_ids)
        with torch.no_grad():
            for _ in range(64):
                ctx = curr_ids[-2048:]
                inp = torch.tensor([ctx], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(inp)
                    next_token = torch.argmax(out.logits[0, -1, :]).item()
                curr_ids.append(next_token)
                if next_token == sp.eos_id():
                    break
        gen_text = sp.decode(curr_ids)
        samples.append({"prompt": p, "completion": gen_text})

    print(f"Writing samples to {SAMPLES_MD}...")
    with open(SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# Phase 5B Data Mixture Ablation: Sample Generations\n\n")
        f.write("Generated using greedy decoding (`temperature = 0.0`, `max_new_tokens = 64`) from decayed 250M-token checkpoints.\n\n")
        for p_idx, prompt in enumerate(PROMPTS):
            category = "Standard Prompt" if p_idx < 10 else "Basic Grammar Prompt"
            f.write(f"### Prompt {p_idx+1} ({category})\n")
            f.write(f"> **Prompt**: `{prompt}`\n\n")
            f.write(f"**Run_A (100% FineWeb-Edu)**:\n")
            f.write(f"```text\n{samples[p_idx]['completion']}\n```\n\n")
            f.write("---\n\n")

    print("Done!")

if __name__ == "__main__":
    main()
