import matplotlib.pyplot as plt
import pandas as pd

# Standardize CSV
clean_rows = []
with open('results/phase5b_curves.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

header = "run_id,run_name,step,tokens_m,wall_sec,active_sec,lr,train_loss\n"

for line in lines[1:]:
    line = line.strip()
    if not line or line.startswith('run_id'):
        continue
    parts = line.split(',')
    if len(parts) == 7:
        # Run_A format: run_id, run_name, step, tokens_m, elapsed_sec, lr, train_loss
        r_id, r_name, step, tok_m, el_sec, lr, loss = parts
        clean_rows.append(f"{r_id},{r_name},{step},{tok_m},{el_sec},{el_sec},{lr},{loss}\n")
    elif len(parts) == 8:
        clean_rows.append(line + "\n")

with open('results/phase5b_curves.csv', 'w', encoding='utf-8') as f:
    f.write(header)
    f.writelines(clean_rows)

print(f"Cleaned {len(clean_rows)} curve entries in results/phase5b_curves.csv")

# Plot Loss vs Tokens
df = pd.read_csv('results/phase5b_curves.csv')

plt.figure(figsize=(10, 6), dpi=150)
colors = {"Run_A": "#1f77b4", "Run_B": "#ff7f0e", "Run_C": "#2ca02c", "Run_D": "#d62728"}
labels = {
    "Run_A": "Run A: 100% FineWeb-Edu",
    "Run_B": "Run B: 100% Cosmopedia v2",
    "Run_C": "Run C: Default Blend (45% Cosmo, 45% FW-Edu, 10% Tiny)",
    "Run_D": "Run D: Simpler Blend (40% Tiny, 40% Cosmo, 20% FW-Edu)",
}

for run_id in ["Run_A", "Run_B", "Run_C", "Run_D"]:
    sub = df[df["run_id"] == run_id]
    if len(sub) > 0:
        plt.plot(sub["tokens_m"], sub["train_loss"], label=labels[run_id], color=colors[run_id], linewidth=2.0)

plt.axvline(x=225, color="gray", linestyle="--", alpha=0.7, label="WSD Cosine Decay Start (225M tokens)")
plt.title("Phase 5B Data Mixture Ablation: Training Loss vs. Tokens (6L_768H_1KV)", fontsize=13, fontweight="bold")
plt.xlabel("Tokens Trained (Millions)", fontsize=11)
plt.ylabel("Training Loss (Cross-Entropy)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("results/phase5b_loss_vs_tokens.png")
print("Saved plot to results/phase5b_loss_vs_tokens.png")
