"""
Step 1 Housekeeping:
1. Label all pre-fix results as 'pre-fix (double-shift)' and mark noise floors provisional.
2. Reconcile head-to-head delta in plan and summary.
3. Fix perplexity ratio math.
4. Weaken Run A single-prompt code claim.
"""

import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# 1. Update CSVs to label pre-fix runs
csv_files_to_tag = [
    "lr_sweep.csv",
    "lr_sweep_extended_summary.csv",
    "trials_1hr_summary.csv",
    "head_to_head_summary.csv",
    "head_to_head_gap_table.csv",
    "noise_estimate.csv",
]

for fname in csv_files_to_tag:
    fpath = RESULTS_DIR / fname
    if fpath.exists():
        try:
            df = pd.read_csv(fpath)
            if "training_status" not in df.columns:
                df["training_status"] = "pre-fix (double-shift)"
                df.to_csv(fpath, index=False)
                print(f"Tagged {fname} with 'pre-fix (double-shift)'")
        except Exception as e:
            print(f"Skipping {fname}: {e}")

# Update noise_report.txt to mark provisional
noise_report_path = RESULTS_DIR / "noise_report.txt"
if noise_report_path.exists():
    with open(noise_report_path, "r", encoding="utf-8") as f:
        text = f.read()
    if "PROVISIONAL" not in text:
        header = "[NOTE: PROVISIONAL VALUES - MEASURED PRE-FIX UNDER DOUBLE-SHIFT BUG]\n"
        with open(noise_report_path, "w", encoding="utf-8") as f:
            f.write(header + text)
        print("Updated noise_report.txt with PROVISIONAL label")

print("Housekeeping CSV updates complete!")
