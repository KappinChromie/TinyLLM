# Phase 5B Data Mixture Ablation Report

## 1. Executive Summary & Results Table

- **Model Architecture**: `6L_768H_1KV` (42.28M parameters, locked winning shape)
- **Volume per Run**: 250M tokens (3,815 optimizer steps of 65,536 tokens)
- **Schedule**: Flat WSD (50-step warmup, flat at $\eta_0 = 1\times 10^{-3}$ to step 3,434, 381-step cosine decay to $0.1\times \eta_0$)
- **Noise Floor Reference**: $\Delta_{\text{val}} = 0.0076$ (validation spread), $\Delta_{\text{train}} = 0.0254$ (training spread)

| Run ID | Mixture Description | Micro BS | Wiki Neutral (CE) | Wiki (BPB) | Cosmo (CE) | FineWeb (CE) | TinyStories (CE) | Active Time | Wall Time | Cooling Pauses | Active Tok/s |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run_A** | 100% FineWeb-Edu | 8 | **2.8349** | 0.9980 | 2.8724 | 2.7818 | 3.0167 | 58.0m | 58.0m | 0 | 71,851 |
| **Run_B** | 100% Cosmopedia v2 | 4 | **3.3572** | 1.1819 | 2.0084 | 3.3926 | 2.6202 | 62.3m | 90.7m | 339 | 66,919 |
| **Run_C** | Default Blend (45% Cosmo, 45% FW-Edu, 10% Tiny) | 4 | **2.9364** | 1.0338 | 2.1149 | 2.8695 | 1.6175 | 62.2m | 83.9m | 259 | 66,997 |
| **Run_D** | Simpler-Text Blend (40% Tiny, 40% Cosmo, 20% FW-Edu) | 4 | **3.0433** | 1.0714 | 2.1608 | 2.9948 | 1.3661 | 57.2m | 88.3m | 371 | 72,836 |

## 2. Decision Analysis on Neutral English Wikipedia Benchmark

The neutral held-out set (`val_wikipedia_neutral`, ~19.35M tokens) is the primary decision benchmark, having 0% training contamination with any candidate slice.

Gaps against default blend (Run C) on neutral English Wikipedia:

- **Run_A (100% FineWeb-Edu)**: $\Delta = -0.1015$ CE vs Run C (Decisive Win (13.4x Val Noise Floor))
- **Run_B (100% Cosmopedia v2)**: $\Delta = +0.4208$ CE vs Run C (Inferior (+55.4x Val Noise Floor))
- **Run_C (Default Blend (45% Cosmo, 45% FW-Edu, 10% Tiny))**: $\Delta = +0.0000$ CE vs Run C (Within Noise Floor)
- **Run_D (Simpler-Text Blend (40% Tiny, 40% Cosmo, 20% FW-Edu))**: $\Delta = +0.1069$ CE vs Run C (Inferior (+14.1x Val Noise Floor))

## 3. Qualitative Evaluation & Fluency Analysis

- **Run A (100% FineWeb-Edu)**: Achieves the lowest perplexity on factual, educational, and encyclopedic text (2.8349 CE on Wikipedia, 2.7818 CE on FineWeb). However, on open-ended creative story prompts, it lacks narrative structure and tends to derail into dry administrative/encyclopedic prose.
- **Run B (100% Cosmopedia v2)**: Excels at structured formatting (Markdown headers, Python code block syntax, textbook definitions). However, it suffers massive domain collapse on neutral web text (3.3572 CE on Wikipedia, +0.5223 CE worse than Run A).
- **Run C (Default Blend: 45% Cosmo, 45% FW-Edu, 10% Tiny)**: Strikingly balanced performance. It achieves strong factual grounding (2.9364 CE on Wikipedia), while adopting Cosmopedia's textbook structure and TinyStories' syntactic narrative flow. Sentences and dialogues are natural and complete.
- **Run D (Simpler-Text Blend: 40% Tiny, 40% Cosmo, 20% FW-Edu)**: Produces the most fluent and coherent children's stories and simple dialogues, but pays a substantial penalty on neutral encyclopedic knowledge (3.0433 CE on Wikipedia, +0.2084 CE worse than Run A, +0.1069 CE worse than Run C).
