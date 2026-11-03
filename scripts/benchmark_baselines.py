import subprocess
import re
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TOOLS_DIR = ROOT / "tools"
LLAMA_BENCH = TOOLS_DIR / "llama-bin" / "llama-bench.exe"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
BASELINE_CSV = RESULTS_DIR / "baseline.csv"

def get_gpu_telemetry():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "temp_c": float(parts[0]),
            "power_w": float(parts[1]),
            "clock_mhz": float(parts[2]),
        }
    except Exception as e:
        return {"temp_c": None, "power_w": None, "clock_mhz": None}

def run_bench(model_path: Path):
    cmd = [
        str(LLAMA_BENCH),
        "-m", str(model_path),
        "-ngl", "99",
        "-fa", "1",
        "-p", "512",
        "-n", "256",
        "-r", "5",
        "-o", "csv"
    ]
    pre_gpu = get_gpu_telemetry()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    post_gpu = get_gpu_telemetry()
    
    # parse CSV lines from llama-bench output
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("build_commit"):
            header_idx = i
            break
            
    if header_idx == -1:
        raise RuntimeError(f"Could not find CSV header in llama-bench output:\n{res.stdout}")
        
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    
    pp_row = next((r for r in rows if r["n_prompt"] == "512" and r["n_gen"] == "0"), None)
    tg_row = next((r for r in rows if r["n_prompt"] == "0" and r["n_gen"] == "256"), None)
    
    return {
        "model_file": model_path.name,
        "model_type": rows[0]["model_type"],
        "model_size_mb": round(int(rows[0]["model_size"]) / (1024 * 1024), 2),
        "n_params_m": round(int(rows[0]["model_n_params"]) / 1e6, 2),
        "pp_ts": float(pp_row["avg_ts"]) if pp_row else None,
        "pp_ts_std": float(pp_row["stddev_ts"]) if pp_row else None,
        "tg_ts": float(tg_row["avg_ts"]) if tg_row else None,
        "tg_ts_std": float(tg_row["stddev_ts"]) if tg_row else None,
        "gpu_temp_c": post_gpu["temp_c"],
        "gpu_power_w": post_gpu["power_w"],
        "gpu_clock_mhz": post_gpu["clock_mhz"],
    }

def main():
    models = [
        MODELS_DIR / "smollm2-135m-f16.gguf",
        MODELS_DIR / "smollm2-135m-Q8_0.gguf",
        MODELS_DIR / "smollm2-135m-Q4_0.gguf",
        MODELS_DIR / "qwen3-0.6b-f16.gguf",
        MODELS_DIR / "qwen3-0.6b-Q8_0.gguf",
        MODELS_DIR / "qwen3-0.6b-Q4_0.gguf",
    ]
    
    results = []
    for m in models:
        print(f"Benchmarking {m.name}...")
        data = run_bench(m)
        print(f"  PP: {data['pp_ts']:.1f} +/- {data['pp_ts_std']:.1f} t/s | TG: {data['tg_ts']:.1f} +/- {data['tg_ts_std']:.1f} t/s | GPU: {data['gpu_temp_c']}C, {data['gpu_power_w']}W")
        results.append(data)
        
    with open(BASELINE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nWrote baseline results to {BASELINE_CSV}")

if __name__ == "__main__":
    main()
