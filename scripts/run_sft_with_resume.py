"""
Auto-resume wrapper for Step 4 Main SFT Run
- Runs scripts/train_sft_main.py --resume
- Retries up to 2 times on non-divergence crash
- Never retries after numerical divergence (loss explosion or NaN)
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_sft_main.py"

MAX_RETRIES = 2

def main():
    retries = 0
    cmd = [sys.executable, str(SCRIPT), "--resume"]
    
    while retries <= MAX_RETRIES:
        print(f"\n{'='*70}\n[WRAPPER] Launching Main SFT Run (Attempt {retries + 1}/{MAX_RETRIES + 1})\n{'='*70}", flush=True)
        res = subprocess.run(cmd)
        
        if res.returncode == 0:
            print("\n[WRAPPER] Main SFT run completed successfully with code 0!", flush=True)
            sys.exit(0)
            
        print(f"\n[WRAPPER WARNING] Main SFT run exited with non-zero code {res.returncode}.", flush=True)
        retries += 1
        if retries > MAX_RETRIES:
            print(f"[WRAPPER FATAL] Maximum retries ({MAX_RETRIES}) reached. Aborting.", flush=True)
            sys.exit(res.returncode)
            
        print(f"[WRAPPER] Attempting auto-resume retry {retries}/{MAX_RETRIES} in 5 seconds...", flush=True)
        import time
        time.sleep(5)

if __name__ == "__main__":
    main()
