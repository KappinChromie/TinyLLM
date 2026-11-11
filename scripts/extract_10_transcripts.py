"""
Extracts 10 representative verbatim transcripts (Best, Typical, Worst) from
results/chat_eval_samples.md to produce results/chat_10_transcripts.md.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_MD = ROOT / "results" / "chat_eval_samples.md"
OUT_MD = ROOT / "results" / "chat_10_transcripts.md"

with open(SAMPLES_MD, "r", encoding="utf-8") as f:
    text = f.read()

# Parse sections
sections = text.split("## Prompt ")
parsed = {}

for s in sections[1:]:
    lines = s.split("\n")
    header = lines[0].strip() # e.g. "greet_01 (greetings)"
    pid = header.split()[0]
    cat = header.split("(")[1].replace(")", "").strip() if "(" in header else ""
    
    # Extract Prompt
    p_match = re.search(r'\*\*Prompt\*\*:\n```\n(.*?)\n```', s, re.DOTALL)
    prompt_txt = p_match.group(1).strip() if p_match else ""
    
    # Extract Chat Model Greedy
    c_match = re.search(r'### Chat Model \(Greedy\):\n(.*?)\n\n### Base Model', s, re.DOTALL)
    chat_greedy = c_match.group(1).strip() if c_match else ""
    
    parsed[pid] = {
        "id": pid,
        "category": cat,
        "prompt": prompt_txt,
        "chat_greedy": chat_greedy,
    }

# 10 Selected Transcripts
selected = [
    # Best (Coherent, correct turn ending, polite, follows instructions)
    {
        "label": "Best",
        "pid": "greet_01",
        "name": "Greeting & Conversational Turn Termination",
        "commentary": "Emits `<|end|>` cleanly after a polite greeting without infinite generation or looping."
    },
    {
        "label": "Best",
        "pid": "fact_09",
        "name": "Simple Fact Retrieval (France Capital)",
        "commentary": "Accurately identifies Paris and terminates turn cleanly with `<|end|>`."
    },
    {
        "label": "Best",
        "pid": "rewrite_29",
        "name": "Polite Rewriting Instruction",
        "commentary": "Successfully converts demanding phrasing into a polite, diplomatic assistant tone."
    },
    {
        "label": "Best",
        "pid": "system_45",
        "name": "System-Prompt Persona Following (Travel Guide)",
        "commentary": "Adopts the travel guide persona, recommending historic sites in Rome with polite conversational flow."
    },
    
    # Typical (Topic relevance preserved, but slight syntactic drift or generic answers)
    {
        "label": "Typical",
        "pid": "multiturn_51",
        "name": "Multi-Turn Conversation (3-Turn Pet Discussion)",
        "commentary": "Maintains context across 3 conversational turns about caring for beginner fish."
    },
    {
        "label": "Typical",
        "pid": "format_24",
        "name": "Bullet-Point List Following (Exercise Benefits)",
        "commentary": "Provides reasons for regular exercise in list format, though with modest syntactic drift."
    },
    {
        "label": "Typical",
        "pid": "summary_37",
        "name": "One-Sentence Summarization (Amazon Rainforest)",
        "commentary": "Captures the core topic of the Amazon basin, staying reasonably concise."
    },
    
    # Worst (Factual error, sentence loop, or long-context drift)
    {
        "label": "Worst",
        "pid": "fact_11",
        "name": "Factual Hallucination / Arithmetic (Leap Year Days)",
        "commentary": "42M parameter capacity limit: hallucinates the number of days in a leap year rather than retrieving 366."
    },
    {
        "label": "Worst",
        "pid": "format_19",
        "name": "Sentence Repetition / Looping",
        "commentary": "Gets caught repeating phrases about the color blue instead of producing exactly one clean sentence."
    },
    {
        "label": "Worst",
        "pid": "long_63",
        "name": "Long-Context (30,000 Tokens) Thematic Overreach",
        "commentary": "At 30k context depth, attends to local syntactic cues but misses the global narrative theme."
    },
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("# Step 5: 10 Verbatim Transcripts of TinyLLM Chat Model (42.28M)\n\n")
    f.write("Verbatim evaluation outputs across 10 representative transcripts covering **Best**, **Typical**, and **Worst** capabilities of the 42M chat model.\n\n---\n\n")
    
    for item in selected:
        pid = item["pid"]
        data = parsed.get(pid, {"prompt": "N/A", "chat_greedy": "N/A"})
        has_end = "<|end|>" in data["chat_greedy"]
        
        f.write(f"## [{item['label']}] {item['name']} (`{pid}`)\n\n")
        f.write("**Conversation Context**:\n```\n")
        f.write(data["prompt"] + "\n```\n\n")
        f.write("**Model Response (Greedy)**:\n```\n")
        f.write(data["chat_greedy"] + "\n```\n\n")
        f.write(f"- **Emitted `<|end|>`**: {'Yes' if has_end else 'No'}\n")
        f.write(f"- **Assessment & Failure Analysis**: {item['commentary']}\n\n")
        f.write("---\n\n")

print(f"Saved 10 verbatim transcripts to {OUT_MD}")
