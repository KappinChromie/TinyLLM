"""
Plotting Script for Stage A Training and Hourly Evaluations
Generates:
1. results/stageA_train_loss_curve.png: Training loss vs Tokens seen (M) & Optimizer steps, with active speed & temperature.
2. results/stageA_hourly_eval_curves.png: Cross-Entropy loss & Bits-Per-Byte (BPB) progression across all 4 held-out distributions.
Copies plots to artifact directory for presentation.
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = Path("/mnt/c/Users/luche/.gemini/antigravity/brain/4108e88f-fd9b-49f3-bf40-12188253dac1")

TRAIN_LOG_CSV = RESULTS_DIR / "stageA_train_log.csv"
HOURLY_EVAL_CSV = RESULTS_DIR / "stageA_hourly_eval.csv"

def plot_stage_a():
    df_train = pd.read_csv(TRAIN_LOG_CSV)
    df_eval = pd.read_csv(HOURLY_EVAL_CSV)

    # 1. Training Loss Curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_xlabel("Tokens Processed (Millions)", fontsize=12, labelpad=8)
    ax1.set_ylabel("Training Loss (Cross-Entropy)", color=color, fontsize=12)
    ax1.plot(df_train["tokens_m"], df_train["train_loss"], color=color, linewidth=2, label="Stage A Training Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis for learning rate
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Learning Rate", color=color, fontsize=12)
    ax2.plot(df_train["tokens_m"], df_train["lr"], color=color, linestyle=":", linewidth=1.5, label="Learning Rate (WSD Flat)")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 2.5e-3)

    plt.title("Stage A Training Run: Loss vs Tokens (6L_768H_1KV, 2.15B Tokens)", fontsize=14, pad=12)
    fig.tight_layout()
    train_plot_path = RESULTS_DIR / "stageA_train_loss_curve.png"
    plt.savefig(train_plot_path, dpi=200)
    plt.close()
    print(f"Saved {train_plot_path}")

    # 2. Hourly Evaluations (CE Loss and BPB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Cross-Entropy Loss
    ax1.plot(df_eval["tokens_m"], df_eval["val_wiki_ce"], marker="o", linewidth=2, label="Wikipedia Neutral (Held-out)")
    ax1.plot(df_eval["tokens_m"], df_eval["val_fineweb_ce"], marker="s", linewidth=2, label="FineWeb-Edu Val (60% Mix)")
    ax1.plot(df_eval["tokens_m"], df_eval["val_cosmo_ce"], marker="^", linewidth=2, label="Cosmopedia Val (30% Mix)")
    ax1.plot(df_eval["tokens_m"], df_eval["val_tinystories_ce"], marker="d", linewidth=2, label="TinyStories Val (10% Mix)")
    ax1.set_title("Hourly Validation Cross-Entropy Loss", fontsize=13)
    ax1.set_xlabel("Tokens Processed (Millions)", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Bits-Per-Byte (BPB)
    ax2.plot(df_eval["tokens_m"], df_eval["val_wiki_bpb"], marker="o", linewidth=2, label="Wikipedia Neutral (Held-out)")
    ax2.plot(df_eval["tokens_m"], df_eval["val_fineweb_bpb"], marker="s", linewidth=2, label="FineWeb-Edu Val")
    ax2.plot(df_eval["tokens_m"], df_eval["val_cosmo_bpb"], marker="^", linewidth=2, label="Cosmopedia Val")
    ax2.plot(df_eval["tokens_m"], df_eval["val_tinystories_bpb"], marker="d", linewidth=2, label="TinyStories Val")
    ax2.set_title("Hourly Validation Bits-Per-Byte (BPB)", fontsize=13)
    ax2.set_xlabel("Tokens Processed (Millions)", fontsize=11)
    ax2.set_ylabel("Bits Per Byte (BPB)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.suptitle("Stage A Hourly Checkpoint Evaluations Across 4 Distributions", fontsize=15, y=0.98)
    fig.tight_layout()
    eval_plot_path = RESULTS_DIR / "stageA_hourly_eval_curves.png"
    plt.savefig(eval_plot_path, dpi=200)
    plt.close()
    print(f"Saved {eval_plot_path}")

    # Copy to artifacts directory
    if ARTIFACT_DIR.exists():
        shutil.copyfile(train_plot_path, ARTIFACT_DIR / "stageA_train_loss_curve.png")
        shutil.copyfile(eval_plot_path, ARTIFACT_DIR / "stageA_hourly_eval_curves.png")
        print(f"Copied plots to {ARTIFACT_DIR}")

if __name__ == "__main__":
    plot_stage_a()
