import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("results", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Stage C Training Loss Curve
df_train = pd.read_csv("results/stageC_train_log.csv")

plt.figure(figsize=(10, 5), dpi=150)
plt.plot(df_train['tokens_m'], df_train['train_loss'], label='Stage C Training Loss (Smoothed/Step)', color='#1f77b4', alpha=0.7, lw=1.5)
# Add rolling average
if len(df_train) > 5:
    rolling = df_train['train_loss'].rolling(5, min_periods=1).mean()
    plt.plot(df_train['tokens_m'], rolling, label='5-Step Rolling Mean', color='#004080', lw=2.2)

# Demarcate C1 vs C2
plt.axvline(x=57.28, color='#d62728', linestyle='--', lw=1.8, label='Step 437: C1 -> C2 Transition (Cosine Decay Begins)')
plt.text(28, 2.7, 'Sub-stage C1 (Flat LR 5e-4)\n75% PG19 / 25% Replay', ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', edgecolor='#1f77b4'))
plt.text(85, 2.7, 'Sub-stage C2 (Cosine -> 1e-4)\n50% PG19 / 50% Short Text', ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', edgecolor='#d62728'))

plt.title('Stage C Training Loss vs Processed Tokens (32k Context)', fontsize=13, fontweight='bold')
plt.xlabel('Cumulative Stage C Tokens (Millions)', fontsize=11)
plt.ylabel('Training Loss (Cross-Entropy)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("results/stageC_train_loss_curve.png")
plt.close()

# 2. Cross-Stage Milestone Validation Curves (Stage A -> B -> C1 -> C Decayed)
# Data points
stages = ['Stage A\n(Undecayed 2k)', 'Stage B\n(Undecayed 8k)', 'Stage C1\n(Undecayed 32k)', 'Stage C\n(Decayed 32k)']
# Exact values from CSVs:
# Stage A: Wiki 2.6787, FW 2.6303, Cosmo 1.9157, Tiny 1.4544
# Stage B: Wiki 2.3774, FW 2.5506, Cosmo 1.8624, Tiny 1.6055
# Stage C1: Wiki 2.4666, FW 2.6305, Cosmo 1.8831, Tiny 1.4232
# Stage C Decayed: Wiki 2.4466, FW 2.5966, Cosmo 1.8406, Tiny 1.3887

wiki_vals = [2.6787, 2.3774, 2.4666, 2.4466]
fw_vals = [2.6303, 2.5506, 2.6305, 2.5966]
cosmo_vals = [1.9157, 1.8624, 1.8831, 1.8406]
tiny_vals = [1.4544, 1.6055, 1.4232, 1.3887]

plt.figure(figsize=(10, 6), dpi=150)
plt.plot(stages, wiki_vals, marker='o', lw=2.2, color='#2ca02c', label='Wikipedia Neutral 2k')
plt.plot(stages, fw_vals, marker='s', lw=2.2, color='#1f77b4', label='FineWeb-Edu 2k')
plt.plot(stages, cosmo_vals, marker='^', lw=2.2, color='#ff7f0e', label='Cosmopedia v2 2k')
plt.plot(stages, tiny_vals, marker='d', lw=2.5, color='#d62728', label='TinyStories 2k (Replay Target)')

# Annotate TinyStories recovery
plt.annotate('Stage B drift: +0.1511', xy=(1, 1.6055), xytext=(0.7, 1.70),
             arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color='#d62728')
plt.annotate('Drift arrested & reversed: 1.3887\n(-0.0657 vs Stage A!)', xy=(3, 1.3887), xytext=(2.2, 1.25),
             arrowprops=dict(facecolor='#2ca02c', shrink=0.05, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color='#006600')

plt.title('Validation Loss Across Training Milestones (Zero Regression & Replay Recovery)', fontsize=13, fontweight='bold')
plt.xlabel('Training Milestone', fontsize=11)
plt.ylabel('Held-out Cross-Entropy Loss (2k Window)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
plt.savefig("results/stageC_milestone_eval_curves.png")
plt.close()

# 3. Position Bucket Loss Curve up to 32k
buckets = ['0-1k', '1k-2k', '2k-4k', '4k-8k', '8k-16k', '16k-32k']
# Values from stageC_eval.csv:
# 2.8139, 2.6664, 2.5388, 2.5705, 2.5322, 2.5985
bucket_losses = [2.8139, 2.6664, 2.5388, 2.5705, 2.5322, 2.5985]

plt.figure(figsize=(9, 5), dpi=150)
bars = plt.bar(buckets, bucket_losses, color='#4575b4', width=0.55, edgecolor='black', alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylim(2.2, 3.0)
plt.title('Stage C Final Decayed Model: Position-Bucket Loss on PG-19 (Up to 32k Context)', fontsize=13, fontweight='bold')
plt.xlabel('Context Window Position Bucket', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.axhline(y=2.5985, color='gray', linestyle=':', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/stageC_position_bucket_curve.png")
plt.close()

# 4. Decode Speed Benchmark Curve (0k, 8k, 16k, 32k)
df_speed = pd.read_csv("results/stageC_decode_speed.csv")

plt.figure(figsize=(9, 5), dpi=150)
markers = {'F16': 'o', 'Q8_0': 's', 'Q4_0': '^'}
colors = {'F16': '#1f77b4', 'Q8_0': '#2ca02c', 'Q4_0': '#ff7f0e'}

for q in ['F16', 'Q8_0', 'Q4_0']:
    sub = df_speed[df_speed['quant'] == q].sort_values('context_depth')
    plt.errorbar(sub['context_depth'] / 1024, sub['decode_tok_s'], yerr=sub['decode_err_tok_s'],
                 label=f'{q} ({sub["decode_tok_s"].iloc[-1]:.1f} t/s @ 32k)',
                 marker=markers[q], color=colors[q], lw=2.2, capsize=4)

plt.axhline(y=1000, color='red', linestyle='--', lw=1.5, label='1,000 tok/s Target')
plt.title('Final Inference Speed vs Context Depth (llama.cpp, RTX 3070 Ti, Flash Attention)', fontsize=13, fontweight='bold')
plt.xlabel('KV Cache Context Depth (k Tokens)', fontsize=11)
plt.ylabel('Generation Speed (tokens / second)', fontsize=11)
plt.xticks([0, 8, 16, 32], ['0k', '8k', '16k', '32k'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("results/stageC_decode_speed.png")
plt.close()

print("All Stage C plots generated successfully!")
