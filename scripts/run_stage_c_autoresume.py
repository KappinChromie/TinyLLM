"""
Stage C Auto-Resume Execution Wrapper
- Monitors train_stage_c.py execution.
- Automatically retries on non-divergence failure up to 2 times (3 attempts total).
- Exits immediately if divergence or loss spike aborts.
"""

import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_stage_c.py"

def main():
    theta = sys.argv[1] if len(sys.argv) > 1 else "1000000"
    max_retries = 2
    attempts = 0
    print("=== Stage C Auto-Resume Wrapper Initialized ===", flush=True)
    
    while attempts <= max_retries:
        attempts += 1
        print(f"\n[WRAPPER] Launching Stage C Training Process (Attempt {attempts}/{max_retries+1}) with RoPE theta {theta}...", flush=True)
        t0 = time.time()
        
        proc = subprocess.run([
            sys.executable, str(SCRIPT), "--rope-theta", str(theta)
        ])
        
        retcode = proc.returncode
        elapsed = time.time() - t0
        print(f"[WRAPPER] Stage C Process exited with code {retcode} after {elapsed:.1f}s.", flush=True)
        
        if retcode == 0:
            print("[WRAPPER] Stage C completed successfully (or cleanly reached milestone/rule checkpoint)!", flush=True)
            break
        else:
            print(f"[WRAPPER] Process crashed/failed with code {retcode}.", flush=True)
            if attempts > max_retries:
                print(f"[WRAPPER] Maximum retry limit ({max_retries}) reached. Halting.", flush=True)
                sys.exit(retcode)
            print("[WRAPPER] Waiting 10 seconds before resuming from latest rolling checkpoint...", flush=True)
            time.sleep(10)

if __name__ == "__main__":
    main()
