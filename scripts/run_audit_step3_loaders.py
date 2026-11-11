import os
import sys
import platform
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"

def test_loader_path(name, code_str, max_idx, sample_fn, num_samples=10_000_000):
    print(f"\nTesting {name} (max_idx: {max_idx:,})...", flush=True)
    try:
        samples = sample_fn(max_idx, num_samples)
        min_val = int(np.min(samples))
        max_val = int(np.max(samples))
        outside = bool(min_val < 0 or max_val > max_idx)
        count_gt_2_31 = int(np.sum(samples >= 2**31))
        
        status = "OK"
        if max_idx >= 2**31 and count_gt_2_31 == 0:
            status = "NEVER_TOUCHED_PAST_2^31"
        elif outside:
            status = "OUT_OF_BOUNDS"
            
        res = {
            "loader_stage": name,
            "sampling_code": code_str,
            "os": platform.system() + " " + platform.release(),
            "numpy_version": np.__version__,
            "max_valid_idx": max_idx,
            "min_sampled": min_val,
            "max_sampled": max_val,
            "count_ge_2^31": count_gt_2_31,
            "outside_valid_range": outside,
            "status": status
        }
        print(f"  Result: min={min_val:,}, max={max_val:,}, ge_2^31={count_gt_2_31:,}, status={status}")
        return res
    except Exception as e:
        print(f"  FAILED with exception: {type(e).__name__}: {e}")
        return {
            "loader_stage": name,
            "sampling_code": code_str,
            "os": platform.system() + " " + platform.release(),
            "numpy_version": np.__version__,
            "max_valid_idx": max_idx,
            "min_sampled": "ERROR",
            "max_sampled": "ERROR",
            "count_ge_2^31": "ERROR",
            "outside_valid_range": "ERROR",
            "status": f"EXCEPTION: {type(e).__name__}: {e}"
        }

def main():
    print("="*75)
    print("STEP 3.1: DATA LOADER RANDOMSTATE & OFFSET OVERFLOW AUDIT")
    print(f"NumPy Version: {np.__version__} | OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print("="*75)

    # Cosmopedia length
    cosmo_path = DATA_DIR / "train_cosmopedia.bin"
    cosmo_tokens = cosmo_path.stat().st_size // 2
    cosmo_max_2k = cosmo_tokens - 2048 - 1
    
    # FineWeb length
    fw_path = DATA_DIR / "train_fineweb_edu.bin"
    fw_tokens = fw_path.stat().st_size // 2
    fw_max_2k = fw_tokens - 2048 - 1
    
    # PG-19 long length
    pg19_path = DATA_DIR / "train_pg19_long.bin"
    pg19_tokens = pg19_path.stat().st_size // 2
    pg19_max_32k = pg19_tokens - 32768 - 1
    
    results = []

    # 1. Phase 5 / Stage A / Decay Branch: Cosmopedia with RandomState.randint(0, max_idx)
    # On Windows NumPy 2.x, RandomState.randint without dtype=np.int64 raises ValueError
    # On Linux (WSL2), C long is int64, so it accepts 64-bit int.
    # If dtype=np.int64 is passed:
    def fn_randomstate_int64(max_idx, n):
        rng = np.random.RandomState(42)
        return rng.randint(0, max_idx, size=n, dtype=np.int64)
    
    results.append(test_loader_path(
        "Stage A & Phase 5 (Cosmopedia with RandomState.randint int64)",
        "rng.randint(0, max_idx, dtype=np.int64)",
        cosmo_max_2k,
        fn_randomstate_int64
    ))

    # Test what raw rng.randint(0, max_idx) without dtype does on current platform
    def fn_randomstate_default(max_idx, n):
        rng = np.random.RandomState(42)
        return rng.randint(0, max_idx, size=n)
    
    results.append(test_loader_path(
        "Stage A & Phase 5 (Cosmopedia with RandomState.randint default dtype)",
        "rng.randint(0, max_idx)",
        cosmo_max_2k,
        fn_randomstate_default
    ))

    # 2. Stage A / Phase 5: FineWeb-Edu with RandomState.randint(0, max_idx)
    results.append(test_loader_path(
        "Stage A & Phase 5 (FineWeb-Edu with RandomState.randint)",
        "rng.randint(0, max_idx)",
        fw_max_2k,
        fn_randomstate_default
    ))

    # 3. Stage B: Long loader (PG-19 8k) with RandomState.randint(0, max_idx)
    pg19_max_8k = pg19_tokens - 8192 - 1
    results.append(test_loader_path(
        "Stage B (PG-19 8k with RandomState.randint)",
        "self.rng_long.randint(0, max_idx)",
        pg19_max_8k,
        fn_randomstate_default
    ))

    # 4. Stage C: Long loader (PG-19 32k) with RandomState.randint(0, max_offset)
    results.append(test_loader_path(
        "Stage C (PG-19 32k with RandomState.randint)",
        "self.rng_long.randint(0, max_offset)",
        pg19_max_32k,
        fn_randomstate_default
    ))

    # 5. SFT LR Trials: default_rng.integers(0, max_idx)
    def fn_default_rng(max_idx, n):
        rng = np.random.default_rng(42)
        return rng.integers(0, max_idx, size=n, dtype=np.int64)
        
    results.append(test_loader_path(
        "SFT LR Trials (default_rng.integers 64-bit)",
        "rng.integers(0, max_idx)",
        cosmo_max_2k,
        fn_default_rng
    ))

    # 6. Main SFT Run: default_rng.integers(0, max_idx)
    results.append(test_loader_path(
        "Main SFT Run (default_rng.integers 64-bit)",
        "rng.integers(0, max_idx)",
        cosmo_max_2k,
        fn_default_rng
    ))

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "step3_data_loader_overflow_audit.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved loader overflow audit table to {out_path}")

if __name__ == "__main__":
    main()
