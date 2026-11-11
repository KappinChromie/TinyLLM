"""
Auto-Resume Wrapper for Stage B Training
- Runs scripts/train_stage_b.py.
- Restarts from latest checkpoint after an unexpected crash (up to 2 retries).
- Does NOT restart if halted due to loss divergence or NaN.
- Logs every restart event to results/stageB_restarts.log.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESTARTS_LOG = RESULTS_DIR / "stageB_restarts.log"
DIVERGENCE_FLAG = RESULTS_DIR / "STOP_DIVERGENCE_STAGEB"
TRAIN_SCRIPT = ROOT / "scripts" / "train_stage_b.py"

PYTHON_BIN = sys.executable

def main():
    print("=== Stage B Auto-Resume Wrapper Initialized ===", flush=True)
    max_retries = 2
    retry_count = 0
    
    while True:
        if DIVERGENCE_FLAG.exists():
            print(f"[WRAPPER] {DIVERGENCE_FLAG} detected! Halting per protocol (no automatic retry on divergence).", flush=True)
            sys.exit(1)
            
        print(f"\n[WRAPPER] Launching Stage B Training Process (Attempt {retry_count + 1}/{max_retries + 1})...", flush=True)
        t_start = time.time()
        
        proc = subprocess.run([PYTHON_BIN, str(TRAIN_SCRIPT)])
        ret_code = proc.returncode
        t_elapsed = time.time() - t_start
        
        if ret_code == 0:
            print(f"\n[WRAPPER] Stage B training completed successfully with return code 0!", flush=True)
            sys.exit(0)
            
        if DIVERGENCE_FLAG.exists():
            print(f"[WRAPPER] Process exited with code {ret_code} due to divergence! Not restarting.", flush=True)
            with open(RESTARTS_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Terminated due to loss divergence (code {ret_code}). No restart.\n")
            sys.exit(ret_code)
            
        retry_count += 1
        with open(RESTARTS_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Crash event: code {ret_code} after {t_elapsed/60:.1f}m. Retry {retry_count}/{max_retries}.\n")
            
        if retry_count > max_retries:
            print(f"\n[!FATAL] Stage B process crashed {retry_count} times! Maximum retries ({max_retries}) exceeded. Halting.", flush=True)
            sys.exit(ret_code)
            
        print(f"\n[!WARNING] Process crashed with exit code {ret_code}. Auto-restarting from checkpoint in 15 seconds (Retry {retry_count}/{max_retries})...", flush=True)
        time.sleep(15)

if __name__ == "__main__":
    main()
