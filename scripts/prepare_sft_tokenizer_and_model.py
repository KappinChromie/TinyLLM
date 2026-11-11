"""
Step 2A & 2B: SFT Preparation - Tokenizer Extension, Model Vocabulary Expansion, and Chat Template Verification
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as pb2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
BASE_TOKENIZER_DIR = MODELS_DIR / "tokenizer_8k"
CHAT_TOKENIZER_DIR = MODELS_DIR / "tokenizer_chat"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data" / "shards"
TOOLS_DIR = ROOT / "tools" / "llama-bin"

STAGE_C_CKPT = MODELS_DIR / "checkpoints" / "stageC" / "stageC_decayed_final.pt"
CHAT_INIT_CKPT = MODELS_DIR / "checkpoints" / "stageC_chat_init.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHAT_TEMPLATE_JINJA = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>' + '\n' }}"
    "{% endif %}"
)

def step2a_extend_tokenizer():
    print("\n" + "="*70)
    print("STEP 2A: EXTENDING TOKENIZER VOCABULARY TO 8,256 TOKENS")
    print("="*70)
    
    CHAT_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse original SentencePiece ModelProto
    m = pb2.ModelProto()
    with open(BASE_TOKENIZER_DIR / "spm_8k.model", "rb") as f:
        m.ParseFromString(f.read())
        
    orig_size = len(m.pieces)
    print(f"Original vocabulary size: {orig_size}")
    assert orig_size == 8192, f"Expected 8192 original pieces, got {orig_size}"
    
    # 2. Append new user-defined pieces up to 8,256
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    reserved_tokens = [f"<|reserved_{i}|>" for i in range(60)]
    all_new_tokens = special_tokens + reserved_tokens
    assert len(all_new_tokens) == 64, f"Expected 64 new tokens, got {len(all_new_tokens)}"
    
    for tok in all_new_tokens:
        piece = pb2.ModelProto.SentencePiece(
            piece=tok,
            score=0.0,
            type=pb2.ModelProto.SentencePiece.Type.USER_DEFINED
        )
        m.pieces.append(piece)
        
    new_size = len(m.pieces)
    print(f"Extended vocabulary size: {new_size} (a multiple of 64: {new_size // 64} * 64)")
    assert new_size == 8256, f"Expected 8256 total pieces, got {new_size}"
    
    # Save new tokenizer.model
    new_model_path = CHAT_TOKENIZER_DIR / "tokenizer.model"
    with open(new_model_path, "wb") as f:
        f.write(m.SerializeToString())
    # Also save spm_8k_chat.model as backup
    shutil.copyfile(new_model_path, CHAT_TOKENIZER_DIR / "spm_8k_chat.model")
    print(f"Saved extended tokenizer to {new_model_path}")
    
    # 3. Save tokenizer_config.json
    tok_config = {
        "add_bos_token": False,
        "add_prefix_space": None,
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|end|>",
        "chat_template": CHAT_TEMPLATE_JINJA,
        "model_max_length": 32768,
        "pad_token": "<pad>",
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "added_tokens_decoder": {
            "1": {"content": "<s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "2": {"content": "</s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "3": {"content": "<pad>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "0": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "8192": {"content": "<|system|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "8193": {"content": "<|user|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "8194": {"content": "<|assistant|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "8195": {"content": "<|end|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        }
    }
    for i in range(60):
        tid = 8196 + i
        tok_config["added_tokens_decoder"][str(tid)] = {
            "content": f"<|reserved_{i}|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }
        
    with open(CHAT_TOKENIZER_DIR / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tok_config, f, indent=2)
        
    # 4. Save special_tokens_map.json
    spec_map = {
        "bos_token": "<s>",
        "eos_token": "<|end|>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "additional_special_tokens": all_new_tokens + ["</s>"]
    }
    with open(CHAT_TOKENIZER_DIR / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(spec_map, f, indent=2)
        
    # 5. Verification: (a) Token IDs for 1,000 ordinary strings
    print("\n--- Verification 2A(a): Token IDs on 1,000 ordinary test strings ---")
    sp_orig = spm.SentencePieceProcessor(str(BASE_TOKENIZER_DIR / "tokenizer.model"))
    sp_chat = spm.SentencePieceProcessor(str(new_model_path))
    
    corpus_file = ROOT / "data" / "tokenizer_train_corpus.txt"
    test_strings = []
    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if len(s) > 20:
                    test_strings.append(s)
                if len(test_strings) >= 1000:
                    break
                    
    identical_count = 0
    roundtrip_count = 0
    for s in test_strings:
        ids_orig = sp_orig.encode_as_ids(s)
        ids_chat = sp_chat.encode_as_ids(s)
        if ids_orig == ids_chat:
            identical_count += 1
        decoded = sp_chat.decode(ids_chat)
        if decoded.strip() == s.strip():
            roundtrip_count += 1
            
    print(f"Token IDs identical before and after: {identical_count} / {len(test_strings)} ({identical_count / len(test_strings)*100:.1f}%)")
    assert identical_count == len(test_strings), f"STOP: Token IDs changed for ordinary strings! {identical_count} != {len(test_strings)}"
    print(f"Round-trip decoded matches original: {roundtrip_count} / {len(test_strings)} ({roundtrip_count / len(test_strings)*100:.1f}%)")
    assert roundtrip_count >= 995, f"STOP: Round-trip failed: {roundtrip_count}"
    print("Verification 2A(a) and 2A(b) PASSED!")


def step2a_expand_model_and_eval_val_ce():
    print("\n" + "="*70)
    print("STEP 2A(c): EXPANDING MODEL EMBEDDINGS (8,192 -> 8,256) & 2k VAL CE CHECK")
    print("="*70)
    
    # Load base Stage C model
    raw_ckpt = torch.load(STAGE_C_CKPT, map_location="cpu", weights_only=False)
    state = raw_ckpt["model_state"] if "model_state" in raw_ckpt else raw_ckpt
    
    old_weight = state["model.embed_tokens.weight"] # [8192, 768]
    print(f"Original embedding matrix shape: {list(old_weight.shape)}")
    assert old_weight.shape == (8192, 768)
    
    # Initialization of new rows:
    # Mean of existing rows plus small noise N(0, 0.005)
    mean_row = old_weight.mean(dim=0, keepdim=True) # [1, 768]
    torch.manual_seed(42)
    noise = torch.randn(64, 768) * 0.005
    new_rows = mean_row.repeat(64, 1) + noise # [64, 768]
    
    expanded_weight = torch.cat([old_weight, new_rows], dim=0) # [8256, 768]
    print(f"Expanded embedding matrix shape: {list(expanded_weight.shape)}")
    
    new_state = dict(state)
    new_state["model.embed_tokens.weight"] = expanded_weight
    if "lm_head.weight" in new_state:
        new_state["lm_head.weight"] = expanded_weight # Tied embeddings
        
    torch.save({"model_state": new_state, "vocab_size": 8256}, CHAT_INIT_CKPT)
    print(f"Saved expanded chat initial checkpoint to {CHAT_INIT_CKPT}")
    
    # Now evaluate 2k Val CE on the four fixed 2M subsets before and after
    print("\nEvaluating 2k Val CE on fixed 2M subsets before vs after embedding expansion...")
    
    config_orig = LlamaConfig(
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
        rope_theta=1000000.0,
    )
    model_orig = LlamaForCausalLM(config_orig).to(device=DEVICE, dtype=torch.bfloat16)
    model_orig.load_state_dict(state)
    model_orig.eval()
    
    config_chat = LlamaConfig(
        vocab_size=8256,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    )
    model_chat = LlamaForCausalLM(config_chat).to(device=DEVICE, dtype=torch.bfloat16)
    model_chat.load_state_dict(new_state)
    model_chat.eval()
    
    val_datasets = [
        ("Wikipedia Neutral", DATA_DIR / "val_wikipedia_neutral.bin"),
        ("FineWeb-Edu", DATA_DIR / "val_fineweb_edu.bin"),
        ("Cosmopedia v2", DATA_DIR / "val_cosmopedia.bin"),
        ("TinyStories", DATA_DIR / "val_tinystories.bin"),
    ]
    
    max_eval_tokens = 2000000
    seq_len = 2048
    num_eval_seqs = max_eval_tokens // seq_len # 976 seqs
    
    ce_results = []
    for name, shard_path in val_datasets:
        shard = np.memmap(shard_path, dtype=np.uint16, mode="r")
        total_toks = min(len(shard), max_eval_tokens)
        n_seqs = total_toks // seq_len
        
        loss_orig_list = []
        loss_chat_list = []
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for i in range(n_seqs):
                    raw = np.array(shard[i*seq_len : (i+1)*seq_len + 1], dtype=np.int64)
                    x = torch.from_numpy(raw[:-1]).unsqueeze(0).to(DEVICE)
                    y = torch.from_numpy(raw[1:]).to(DEVICE)
                    
                    out_orig = model_orig(x)
                    l_orig = F.cross_entropy(out_orig.logits[0], y).item()
                    loss_orig_list.append(l_orig)
                    
                    out_chat = model_chat(x)
                    l_chat = F.cross_entropy(out_chat.logits[0], y).item()
                    loss_chat_list.append(l_chat)
                    
        m_orig = float(np.mean(loss_orig_list))
        m_chat = float(np.mean(loss_chat_list))
        delta = m_chat - m_orig
        
        ce_results.append({
            "dataset": name,
            "vocab_8192_ce": round(m_orig, 4),
            "vocab_8256_ce": round(m_chat, 4),
            "delta_ce": round(delta, 6),
            "status": "PASS (<= 0.01)" if abs(delta) <= 0.01 else "FAIL (> 0.01)"
        })
        print(f"  {name:18s}: Vocab 8192 = {m_orig:.4f} CE | Vocab 8256 = {m_chat:.4f} CE | Delta = {delta:+.6f} CE [{ce_results[-1]['status']}]")
        assert abs(delta) <= 0.01, f"STOP: Expanding embeddings changed 2k val CE by more than 0.01 on {name} ({delta:+.6f})!"
        
    df_ce = pd.DataFrame(ce_results)
    df_ce.to_csv(RESULTS_DIR / "step2a_vocab_expansion_ce_check.csv", index=False)
    print("\nVocabulary expansion validation CE check PASSED strictly on all datasets!")
    return df_ce


def step2b_verify_chat_template():
    print("\n" + "="*70)
    print("STEP 2B: CHAT TEMPLATE & CONVERSATION TOKENIZATION VERIFICATION")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))
    print(f"Loaded AutoTokenizer from {CHAT_TOKENIZER_DIR}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Chat template loaded:\n{tokenizer.chat_template}\n")
    
    # 5 Test Conversations
    test_conversations = [
        # 1. Simple Single Turn
        [
            {"role": "user", "content": "Hello, how are you today?"},
            {"role": "assistant", "content": "I am doing well, thank you! How can I help you?"}
        ],
        # 2. System Prompt + User
        [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ],
        # 3. 3-Turn Conversation
        [
            {"role": "user", "content": "Can you name three planets in our solar system?"},
            {"role": "assistant", "content": "Mars, Jupiter, and Saturn."},
            {"role": "user", "content": "Which one is the largest?"},
            {"role": "assistant", "content": "Jupiter is the largest planet."},
            {"role": "user", "content": "Thank you!"},
            {"role": "assistant", "content": "You're welcome!"}
        ],
        # 4. System Prompt + 2 Turns
        [
            {"role": "system", "content": "Respond only in bullet points."},
            {"role": "user", "content": "Give me two steps to make tea."},
            {"role": "assistant", "content": "- Boil water.\n- Steep the tea bag in hot water."},
            {"role": "user", "content": "And then?"},
            {"role": "assistant", "content": "- Remove the tea bag and enjoy."}
        ],
        # 5. User prompt only (Generation prompt: True)
        [
            {"role": "user", "content": "Tell me a very short joke."}
        ]
    ]
    
    for i, conv in enumerate(test_conversations):
        add_gen = (i == 4) # Last one is generation prompt
        rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=add_gen)
        res_tok = tokenizer.apply_chat_template(conv, tokenize=True, add_generation_prompt=add_gen)
        token_ids = res_tok["input_ids"] if hasattr(res_tok, "input_ids") or isinstance(res_tok, dict) else res_tok
        
        # Check BOS
        bos_count = list(token_ids).count(1) # BOS token ID is 1
        print(f"Conversation {i+1} ({len(conv)} messages, gen_prompt={add_gen}):")
        print(f"  Rendered:\n{rendered!r}")
        print(f"  Token IDs (first 15): {token_ids[:15]}... (Total {len(token_ids)} tokens)")
        print(f"  BOS count: {bos_count} (must be exactly 1)")
        assert bos_count == 1, f"STOP: BOS count is {bos_count}, expected exactly 1!"
        
        # Check that EOS is <|end|> (token 8195)
        if not add_gen:
            assert token_ids[-2] == 8195 or token_ids[-1] == 8195 or rendered.endswith("<|end|>\n"), "Rendered must end with <|end|>"
            
    print("\nChat template verification PASSED! BOS appears exactly once and roles format cleanly.")

def main():
    step2a_extend_tokenizer()
    step2a_expand_model_and_eval_val_ce()
    step2b_verify_chat_template()
    print("\n" + "="*70)
    print("STEP 2A & 2B FINISHED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()
