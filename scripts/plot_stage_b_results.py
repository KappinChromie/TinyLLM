"""
Plotting Script for Stage B Training and Milestone Evaluations
Generates:
1. results/stageB_train_loss_curve.png: Training loss vs Tokens seen (M) & Optimizer steps, with active speed & temperature.
2. results/stageB_milestone_eval_curves.png: 2k Validation CE progression and Position-Bucket loss across milestones.
Copies plots to artifact directory for presentation.
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = Path("/mnt/c/Users/luche/.gemini/antigravity/brain/4108e88f-fd9b-49f3-bf40-12188253dac1")

TRAIN_LOG_CSV = RESULTS_DIR / "stageB_train_log.csv"
EVAL_LOG_CSV = RESULTS_DIR / "stageB_eval.csv"

def plot_stage_b():
    df_train = pd.read_csv(TRAIN_LOG_CSV)
    df_eval = pd.read_csv(EVAL_LOG_CSV)

    # 1. Training Loss Curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_xlabel("Tokens Processed (Millions)", fontsize=12, labelpad=8)
    ax1.set_ylabel("Training Loss (Cross-Entropy)", color=color, fontsize=12)
    ax1.plot(df_train["tokens_m"], df_train["train_loss"], color=color, linewidth=2, label="Stage B Training Loss (8k Context)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis for learning rate
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Learning Rate", color=color, fontsize=12)
    ax2.plot(df_train["tokens_m"], df_train["lr"], color=color, linestyle=":", linewidth=1.5, label="Learning Rate (Flat 0.7e-3)")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1.0e-3)

    plt.title("Stage B Training Run: Loss vs Tokens (8,192 Context, 250.1M Tokens)", fontsize=14, pad=12)
    fig.tight_layout()
    train_plot_path = RESULTS_DIR / "stageB_train_loss_curve.png"
    plt.savefig(train_plot_path, dpi=200)
    plt.close()
    print(f"Saved {train_plot_path}")

    # 2. Milestone Evaluations: 2k Validation CE & Position-Bucket Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2k Cross-Entropy Loss
    ax1.plot(df_eval["tokens_m"], df_eval["val_wiki_2k_ce"], marker="o", linewidth=2, label="Wikipedia Neutral (Held-out)")
    ax1.plot(df_eval["tokens_m"], df_eval["val_fineweb_2k_ce"], marker="s", linewidth=2, label="FineWeb-Edu 2k Val")
    ax1.plot(df_eval["tokens_m"], df_eval["val_cosmo_2k_ce"], marker="^", linewidth=2, label="Cosmopedia v2 2k Val")
    ax1.plot(df_eval["tokens_m"], df_eval["val_tinystories_2k_ce"], marker="d", linewidth=2, label="TinyStories 2k Val")
    ax1.set_title("Stage B 2k Validation Cross-Entropy Progression", fontsize=13)
    ax1.set_xlabel("Tokens Processed (Millions)", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Position-Bucket Loss Progression
    ax2.plot(df_eval["tokens_m"], df_eval["pg19_0_1k"], marker="o", linewidth=2, label="PG-19 [0-1k]")
    ax2.plot(df_eval["tokens_m"], df_eval["pg19_1k_2k"], marker="s", linewidth=2, label="PG-19 [1k-2k]")
    ax2.plot(df_eval["tokens_m"], df_eval["pg19_2k_4k"], marker="^", linewidth=2, label="PG-19 [2k-4k]")
    ax2.plot(df_eval["tokens_m"], df_eval["pg19_4k_8k"], marker="d", linewidth=2, label="PG-19 [4k-8k]")
    ax2.plot(df_eval["tokens_m"], df_eval["wiki_4k_8k"], marker="x", linestyle="--", linewidth=1.5, label="Wiki Long [4k-8k]")
    ax2.plot(df_eval["tokens_m"], df_eval["fineweb_4k_8k"], marker="v", linestyle="--", linewidth=1.5, label="FineWeb Long [4k-8k]")
    ax2.set_title("Stage B 8k Position-Bucket Loss Progression", fontsize=13)
    ax2.set_xlabel("Tokens Processed (Millions)", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    eval_plot_path = RESULTS_DIR / "stageB_milestone_eval_curves.png"
    plt.savefig(eval_plot_path, dpi=200)
    plt.close()
    print(f"Saved {eval_plot_path}")

    # Copy to artifact directory
    if ARTIFACT_DIR.exists():
        shutil.copy(train_plot_path, ARTIFACT_DIR / "stageB_train_loss_curve.png")
        shutil.copy(eval_plot_path, ARTIFACT_DIR / "stageB_milestone_eval_curves.png")
        print(f"Copied plots to {ARTIFACT_DIR}")

if __name__ == "__main__":
    plot_stage_b()
