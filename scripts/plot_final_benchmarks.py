import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("results", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Batched Throughput vs Concurrency Plot
df_batch = pd.read_csv("results/step4_batched_throughput.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

# Short Prompt (128p) scaling
for q, color, marker in [('F16', '#1f77b4', 'o'), ('Q8_0', '#2ca02c', 's'), ('Q4_0', '#d62728', '^')]:
    sub = df_batch[(df_batch['quant'] == q) & (df_batch['suite'] == 'short_128p')]
    ax1.plot(sub['concurrency'], sub['aggregate_tok_s'], marker=marker, lw=2.2, color=color, label=f'{q} (Short 128p/256g)')

ax1.set_title('Aggregate Throughput vs Concurrency (Short Context)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Concurrency (Parallel Streams)', fontsize=10)
ax1.set_ylabel('Aggregate Generation Speed (tokens/sec)', fontsize=10)
ax1.set_xscale('log', base=2)
ax1.set_xticks([1, 8, 32, 64, 128, 256])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.axhline(y=16077, color='#1f77b4', linestyle=':', alpha=0.7)
ax1.annotate('Peak: 16,077 t/s (F16, Conc 128)', xy=(128, 16077), xytext=(32, 17000),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1.2, headwidth=5),
             fontsize=9, fontweight='bold', color='#004080')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

# Throughput across Context Lengths (Q8_0)
sub_q8 = df_batch[df_batch['quant'] == 'Q8_0']
suites = [
    ('short_128p', 'Short Context (128p)', '#1f77b4', 'o'),
    ('context_2k', '2k Context (2048p)', '#2ca02c', 's'),
    ('context_8k', '8k Context (8192p)', '#ff7f0e', '^'),
    ('context_32k', '32k Context (32768p)', '#9467bd', 'd'),
]
for suite, label, color, marker in suites:
    s = sub_q8[sub_q8['suite'] == suite]
    ax2.plot(s['concurrency'], s['aggregate_tok_s'], marker=marker, lw=2.2, color=color, label=label)

ax2.set_title('Q8_0 Aggregate Throughput across Context Depths', fontsize=12, fontweight='bold')
ax2.set_xlabel('Concurrency (Parallel Streams)', fontsize=10)
ax2.set_ylabel('Aggregate Generation Speed (tokens/sec)', fontsize=10)
ax2.set_xscale('log', base=2)
ax2.set_xticks([1, 4, 8, 16, 32, 64, 128, 256])
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.annotate('32k Conc 16: 3,602 t/s\n(VRAM KV: 774 MB)', xy=(16, 3601.63), xytext=(16, 5500),
             arrowprops=dict(facecolor='#9467bd', shrink=0.05, width=1.2, headwidth=5),
             fontsize=9, fontweight='bold', color='#59287a')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig("results/step4_throughput_concurrency.png")
plt.close()

# 2. Single-Stream Decode Speed vs Context Depth
df_single = pd.read_csv("results/step4_single_stream_decode_speed.csv")

plt.figure(figsize=(9, 5), dpi=150)
depths_k = [0, 8, 16, 32]
for q, color, marker in [('F16', '#1f77b4', 'o'), ('Q8_0', '#2ca02c', 's'), ('Q4_0', '#d62728', '^')]:
    sub = df_single[df_single['quant'] == q]
    plt.errorbar(depths_k, sub['decode_tok_s'], yerr=sub['decode_err_tok_s'], marker=marker, lw=2.2, capsize=4, color=color, label=f'{q} (Single-Stream)')

plt.title('Single-Stream Generation Speed vs Context Depth (0k to 32k)', fontsize=13, fontweight='bold')
plt.xlabel('Pre-filled Context Depth (Thousands of Tokens)', fontsize=11)
plt.ylabel('Decode Speed (tokens/sec)', fontsize=11)
plt.xticks([0, 8, 16, 32], ['0k (Pure Gen)', '8,192 (8k)', '16,384 (16k)', '32,768 (32k)'])
plt.ylim(950, 1650)
plt.axhline(y=1000, color='gray', linestyle=':', label='1,000 tok/s Line')
plt.annotate('All quants > 1,068 tok/s at 32k context depth!', xy=(32, 1176), xytext=(18, 1030),
             arrowprops=dict(facecolor='#2ca02c', shrink=0.05, width=1.2, headwidth=5),
             fontsize=10, fontweight='bold', color='#006600')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("results/step4_single_stream_depth.png")
plt.close()

# 3. Context-Use Forest Plot (95% Confidence Intervals)
df_ci = pd.read_csv("results/step2_context_use_statistics.csv")

plt.figure(figsize=(10, 5), dpi=150)
y_pos = np.arange(len(df_ci))
labels = [f"{row['dataset']} ({row['test_type']})" for _, row in df_ci.iterrows()]
means = df_ci['mean_paired_delta'].values
err_low = means - df_ci['ci_95_low'].values
err_high = df_ci['ci_95_high'].values - means

plt.errorbar(means, y_pos, xerr=[err_low, err_high], fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=2.5, capsize=6, markersize=8)
plt.axvline(x=0.0, color='#d62728', linestyle='--', lw=1.8, label='Zero Line (Threshold = 0.0 CE)')
plt.axvline(x=0.02, color='gray', linestyle=':', lw=1.5, label='Loose Threshold (+0.02 CE)')

for i, mean in enumerate(means):
    ci_txt = f"{mean:.4f} [{df_ci['ci_95_low'].iloc[i]:.4f}, {df_ci['ci_95_high'].iloc[i]:.4f}]"
    plt.text(mean, i + 0.22, ci_txt, ha='center', fontsize=9, fontweight='bold', color='#004080')

plt.yticks(y_pos, labels, fontsize=10)
plt.xlabel('Paired Loss Difference: d_i = Loss(Full) - Loss(Short) (Cross-Entropy)', fontsize=11)
plt.title('Context-Use Benefit: 95% Confidence Intervals (Negative = Context Helps)', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("results/step2_context_use_forest_plot.png")
plt.close()

print("Final benchmark plots created successfully!")
