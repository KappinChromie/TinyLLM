import json
import re
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = ROOT / "data" / "chat_eval_prompts.json"
SAMPLES_FILE = ROOT / "results" / "chat_eval_samples.md"
OUT_FILE = ROOT / "results" / "chat_10_transcripts.md"
CHAT_TOKENIZER_DIR = ROOT / "models" / "tokenizer_chat"

def main():
    tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))
    
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)
    prompts_by_id = {p["id"]: p for p in prompts_list}
    
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        samples_text = f.read()

    sections = samples_text.split("## Prompt ")
    parsed = {}
    for s in sections[1:]:
        lines = s.split("\n")
        header = lines[0].strip()
        pid = header.split()[0]
        cat = header.split("(")[1].replace(")", "").strip() if "(" in header else ""
        
        # greedy chat
        c_match = re.search(r'### Chat Model \(Greedy\):\n(.*?)\n\n### Base Model', s, re.DOTALL)
        chat_greedy = c_match.group(1).strip() if c_match else ""
        
        # greedy base
        b_match = re.search(r'### Base Model \(Greedy\):\n(.*?)\n\n### Chat Model Samples', s, re.DOTALL)
        base_greedy = b_match.group(1).strip() if b_match else ""
        
        c_toks = tokenizer.encode(chat_greedy, add_special_tokens=False)
        has_end = ("<|end|>" in chat_greedy) or (8195 in c_toks)
        
        parsed[pid] = {
            "id": pid,
            "category": cat,
            "chat_greedy": chat_greedy,
            "base_greedy": base_greedy,
            "token_len": len(c_toks),
            "has_end": has_end,
            "prompt_obj": prompts_by_id.get(pid, {})
        }
        
    # Sort deterministically by token length, then by ID
    sorted_items = sorted(parsed.values(), key=lambda x: (x["token_len"], x["id"]))
    
    print(f"Total prompts parsed: {len(sorted_items)}")
    print("Length range:", sorted_items[0]["token_len"], "to", sorted_items[-1]["token_len"])
    
    # 10 Selected Transcripts rule:
    # 3 shortest (Ranks 1, 2, 3)
    # 4 median (Ranks 30, 31, 32, 33 - 0-indexed 29, 30, 31, 32)
    # 3 longest (Ranks 61, 62, 63 - 0-indexed 60, 61, 62)
    selected_indices = [0, 1, 2, 29, 30, 31, 32, 60, 61, 62]
    selected_chat = [sorted_items[i] for i in selected_indices]
    
    # 5 Base contrasts (Ranks 1, 16, 32, 48, 63 - 0-indexed 0, 15, 31, 47, 62)
    base_indices = [0, 15, 31, 47, 62]
    selected_base = [sorted_items[i] for i in base_indices]
    
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Audited Chat & Base Model Evaluation Transcripts\n\n")
        f.write("Selection Rule (Deterministic, Not Hand-Picked):\n")
        f.write("- All 63 evaluation prompts sorted deterministically by greedy generation token length (tie-break by prompt ID).\n")
        f.write("- **Extremes (Shortest)**: Ranks 1, 2, 3 (succinct responses)\n")
        f.write("- **Median (Typical)**: Ranks 30, 31, 32, 33 (median response distribution)\n")
        f.write("- **Extremes (Longest)**: Ranks 61, 62, 63 (long / repeating responses hitting 256 tokens)\n")
        f.write("- **Base Contrast**: 5 prompts across the rank spectrum (Ranks 1, 16, 32, 48, 63)\n\n")
        f.write("---\n\n")
        
        f.write("## Part 1: 10 Verbatim Chat Model Transcripts\n\n")
        for rank_num, item in zip([1, 2, 3, 30, 31, 32, 33, 61, 62, 63], selected_chat):
            p_obj = item["prompt_obj"]
            msgs = p_obj.get("messages", [])
            sys_msg = [m["content"] for m in msgs if m["role"] == "system"]
            user_msg = [m["content"] for m in msgs if m["role"] == "user"]
            
            tier = "Shortest Extremes" if rank_num <= 3 else ("Median Distribution" if rank_num <= 33 else "Longest Extremes (Hit 256 Limit)")
            
            f.write(f"### Transcript #{rank_num} (Rank {rank_num}/63, {tier})\n")
            f.write(f"- **Prompt ID**: `{item['id']}`\n")
            f.write(f"- **Category**: `{item['category']}`\n")
            f.write(f"- **Token Length**: {item['token_len']} tokens\n")
            f.write(f"- **Emitted `<|end|>`**: `{item['has_end']}`\n")
            f.write(f"- **Decoding Settings**: Greedy (`temperature=0.0`, `max_new_tokens=256`, `eos_token_id=8195`)\n\n")
            
            if sys_msg:
                f.write(f"**SYSTEM PROMPT**:\n```\n{sys_msg[0]}\n```\n\n")
            f.write(f"**USER INPUT**:\n```\n{user_msg[0] if user_msg else ''}\n```\n\n")
            f.write(f"**CHAT MODEL OUTPUT**:\n```\n{item['chat_greedy']}\n```\n\n")
            f.write("---\n\n")
            
        f.write("## Part 2: 5 Base Model Contrast Transcripts\n\n")
        for rank_num, item in zip([1, 16, 32, 48, 63], selected_base):
            p_obj = item["prompt_obj"]
            msgs = p_obj.get("messages", [])
            sys_msg = [m["content"] for m in msgs if m["role"] == "system"]
            user_msg = [m["content"] for m in msgs if m["role"] == "user"]
            
            f.write(f"### Base Contrast #{rank_num} (Rank {rank_num}/63 Prompt: `{item['id']}`)\n")
            f.write(f"- **Prompt ID**: `{item['id']}`\n")
            f.write(f"- **Category**: `{item['category']}`\n")
            f.write(f"- **Chat Model Output Length**: {item['token_len']} tokens (Emitted `<|end|>`: {item['has_end']})\n")
            f.write(f"- **Base Model Output Length**: 256 tokens (Emitted `<|end|>`: False)\n")
            f.write(f"- **Decoding Settings**: Greedy (`temperature=0.0`, `max_new_tokens=256`)\n\n")
            if sys_msg:
                f.write(f"**SYSTEM PROMPT**:\n```\n{sys_msg[0]}\n```\n\n")
            f.write(f"**USER INPUT**:\n```\n{user_msg[0] if user_msg else ''}\n```\n\n")
            f.write(f"**BASE MODEL OUTPUT**:\n```\n{item['base_greedy']}\n```\n\n")
            f.write("---\n\n")
            
    print(f"Saved 10 chat and 5 base transcripts to {OUT_FILE}")

if __name__ == "__main__":
    main()
