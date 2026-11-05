import pandas as pd

df = pd.read_csv('results/phase5b_summary.csv')

NOISE_FLOOR_VAL = 0.0076   # 2-seed validation spread
NOISE_FLOOR_TRAIN = 0.0254 # 3-seed training spread

ref_row = df[df["run_id"] == "Run_C"]
ref_wiki = ref_row["val_wiki_ce"].values[0]

with open('results/phase5b_report.md', 'w', encoding='utf-8') as f:
    f.write("# Phase 5B Data Mixture Ablation Report\n\n")
    f.write("## 1. Executive Summary & Results Table\n\n")
    f.write("- **Model Architecture**: `6L_768H_1KV` (42.28M parameters, locked winning shape)\n")
    f.write("- **Volume per Run**: 250M tokens (3,815 optimizer steps of 65,536 tokens)\n")
    f.write("- **Schedule**: Flat WSD (50-step warmup, flat at $\\eta_0 = 1\\times 10^{-3}$ to step 3,434, 381-step cosine decay to $0.1\\times \\eta_0$)\n")
    f.write(f"- **Noise Floor Reference**: $\\Delta_{{\\text{{val}}}} = {NOISE_FLOOR_VAL:.4f}$ (validation spread), $\\Delta_{{\\text{{train}}}} = {NOISE_FLOOR_TRAIN:.4f}$ (training spread)\n\n")
    
    f.write("| Run ID | Mixture Description | Micro BS | Wiki Neutral (CE) | Wiki (BPB) | Cosmo (CE) | FineWeb (CE) | TinyStories (CE) | Active Time | Wall Time | Cooling Pauses | Active Tok/s |\n")
    f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for _, r in df.iterrows():
        mbs = int(r['micro_bs'])
        pauses = int(r['num_pauses'])
        active_t = r['active_train_min']
        wall_t = r['total_wall_min']
        tps = r['active_tok_per_sec']
        f.write(f"| **{r['run_id']}** | {r['run_name']} | {mbs} | **{r['val_wiki_ce']:.4f}** | {r['val_wiki_bpb']:.4f} | {r['val_cosmo_ce']:.4f} | {r['val_fineweb_ce']:.4f} | {r['val_tinystories_ce']:.4f} | {active_t:.1f}m | {wall_t:.1f}m | {pauses} | {tps:,.0f} |\n")
    
    f.write("\n## 2. Decision Analysis on Neutral English Wikipedia Benchmark\n\n")
    f.write(f"The neutral held-out set (`val_wikipedia_neutral`, ~19.35M tokens) is the primary decision benchmark, having 0% training contamination with any candidate slice.\n\n")
    f.write("Gaps against default blend (Run C) on neutral English Wikipedia:\n\n")
    for _, r in df.iterrows():
        gap = r['val_wiki_ce'] - ref_wiki
        sig_val = abs(gap) > NOISE_FLOOR_VAL
        sig_train = abs(gap) > NOISE_FLOOR_TRAIN
        sig_str = f"Decisive Win ({abs(gap)/NOISE_FLOOR_VAL:.1f}x Val Noise Floor)" if gap < -NOISE_FLOOR_VAL else (f"Inferior (+{abs(gap)/NOISE_FLOOR_VAL:.1f}x Val Noise Floor)" if gap > NOISE_FLOOR_VAL else "Within Noise Floor")
        f.write(f"- **{r['run_id']} ({r['run_name']})**: $\\Delta = {gap:+.4f}$ CE vs Run C ({sig_str})\n")

    f.write("\n## 3. Qualitative Evaluation & Fluency Analysis\n\n")
    f.write("- **Run A (100% FineWeb-Edu)**: Achieves the lowest perplexity on factual, educational, and encyclopedic text (2.8349 CE on Wikipedia, 2.7818 CE on FineWeb). However, on open-ended creative story prompts, it lacks narrative structure and tends to derail into dry administrative/encyclopedic prose.\n")
    f.write("- **Run B (100% Cosmopedia v2)**: Excels at structured formatting (Markdown headers, Python code block syntax, textbook definitions). However, it suffers massive domain collapse on neutral web text (3.3572 CE on Wikipedia, +0.5223 CE worse than Run A).\n")
    f.write("- **Run C (Default Blend: 45% Cosmo, 45% FW-Edu, 10% Tiny)**: Strikingly balanced performance. It achieves strong factual grounding (2.9364 CE on Wikipedia), while adopting Cosmopedia's textbook structure and TinyStories' syntactic narrative flow. Sentences and dialogues are natural and complete.\n")
    f.write("- **Run D (Simpler-Text Blend: 40% Tiny, 40% Cosmo, 20% FW-Edu)**: Produces the most fluent and coherent children's stories and simple dialogues, but pays a substantial penalty on neutral encyclopedic knowledge (3.0433 CE on Wikipedia, +0.2084 CE worse than Run A, +0.1069 CE worse than Run C).\n")

print("Report re-rendered successfully!")
