import os
import sys
import shutil
import subprocess
import csv
import time
import threading
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SPM_PATH = MODELS_DIR / "spm_placeholders" / "spm_8192.model"
TOOLS_DIR = ROOT / "tools"
LLAMA_BENCH = TOOLS_DIR / "llama-bin" / "llama-bench.exe"
CONVERT_SCRIPT = TOOLS_DIR / "llama.cpp-src" / "convert_hf_to_gguf.py"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "long_context_sweep.csv"

def get_current_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return float(out)
    except Exception:
        return 0.0

def track_peak_vram_and_telemetry(proc):
    peak_vram = 0.0
    while proc.poll() is None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            ).strip()
            v = float(out)
            if v > peak_vram:
                peak_vram = v
        except Exception:
            pass
        time.sleep(0.04)
    return peak_vram

def build_and_export_model(name, layers, hidden, ffn, heads, kv_heads, vocab=8192):
    tmp_dir = ROOT / f"tmp_model_{name}"
    gguf_path = ROOT / f"tmp_{name}.gguf"
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SPM_PATH, tmp_dir / "tokenizer.model")

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=ffn,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=131072,
        tie_word_embeddings=True,
        torch_dtype="float16"
    )
    m = LlamaForCausalLM(cfg).half()
    m.save_pretrained(tmp_dir)
    del m
    torch.cuda.empty_cache()

    if gguf_path.exists():
        gguf_path.unlink()
        
    cmd_conv = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(tmp_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16"
    ]
    subprocess.run(cmd_conv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp_dir)
    return gguf_path

def run_depth_benchmark(gguf_path, depth, repetitions=5):
    cmd_bench = [
        str(LLAMA_BENCH),
        "-m", str(gguf_path),
        "-ngl", "99",
        "-fa", "1",
        "-d", str(depth),
        "-p", "512",
        "-n", "256",
        "-r", str(repetitions),
        "-o", "csv"
    ]
    
    baseline_vram = get_current_vram()
    proc = subprocess.Popen(cmd_bench, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_vram = track_peak_vram_and_telemetry(proc)
    stdout, stderr = proc.communicate()
    
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed (exit {proc.returncode}):\n{stderr}")
        
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("build_commit"):
            header_idx = i
            break
            
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    
    pp_row = next((r for r in rows if r["n_prompt"] == "512" and r["n_gen"] == "0"), None)
    tg_row = next((r for r in rows if r["n_prompt"] == "0" and r["n_gen"] == "256"), None)
    
    # Query final GPU telemetry
    try:
        telemetry = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split(",")
        temp_c = float(telemetry[0].strip())
        power_w = float(telemetry[1].strip())
        clock_mhz = float(telemetry[2].strip())
    except Exception:
        temp_c, power_w, clock_mhz = None, None, None
        
    return {
        "pp_ts": float(pp_row["avg_ts"]) if pp_row else None,
        "pp_ts_std": float(pp_row["stddev_ts"]) if pp_row else None,
        "tg_ts": float(tg_row["avg_ts"]) if tg_row else None,
        "tg_ts_std": float(tg_row["stddev_ts"]) if tg_row else None,
        "baseline_vram_mib": baseline_vram,
        "peak_vram_mib": peak_vram,
        "delta_vram_mib": round(max(0.0, peak_vram - baseline_vram), 1),
        "gpu_temp_c": temp_c,
        "gpu_power_w": power_w,
        "gpu_clock_mhz": clock_mhz,
    }

def main():
    finalists = [
        {"name": "4L_768H", "layers": 4, "hidden": 768, "ffn": 2048, "heads": 12, "kv_heads_list": [2, 1]},
        {"name": "6L_512H", "layers": 6, "hidden": 512, "ffn": 1280, "heads": 8, "kv_heads_list": [2, 1]},
        {"name": "6L_768H", "layers": 6, "hidden": 768, "ffn": 2048, "heads": 12, "kv_heads_list": [2, 1]},
    ]
    
    # Context depths: 0, 2k, 8k, 16k, 32k, and 100k (info-only)
    depths = [0, 2048, 8192, 16384, 32768, 100000]
    
    all_results = []
    
    print("=== STARTING LONG-CONTEXT BENCHMARK (5 repetitions per run) ===")
    
    for f in finalists:
        for kv in f["kv_heads_list"]:
            model_tag = f"{f['name']}_kv{kv}"
            print(f"\nBuilding GGUF model: {model_tag} ({f['layers']}L, {f['hidden']}H, {f['ffn']}FFN, {kv} KV heads)...")
            gguf_path = build_and_export_model(
                model_tag, f["layers"], f["hidden"], f["ffn"], f["heads"], kv, vocab=8192
            )
            file_size_mb = round(gguf_path.stat().st_size / (1024 * 1024), 2)
            
            # If kv == 1, user specifically asked to compare 1 vs 2 at 32k depth.
            # We will test 0, 8k, 32k, and 100k for kv=1 to give a complete curve.
            active_depths = depths if kv == 2 else [0, 8192, 32768, 100000]
            
            d0_tg = None
            for d in active_depths:
                depth_label = f"{d//1000}k" if d >= 1000 else "0"
                print(f"  Benchmarking depth={d} ({depth_label})...", end=" ", flush=True)
                t0 = time.time()
                res = run_depth_benchmark(gguf_path, d, repetitions=5)
                dt = time.time() - t0
                
                if d == 0:
                    d0_tg = res["tg_ts"]
                retention_pct = round((res["tg_ts"] / d0_tg) * 100, 1) if d0_tg else 100.0
                
                row = {
                    "model": f["name"],
                    "layers": f["layers"],
                    "hidden": f["hidden"],
                    "ffn": f["ffn"],
                    "heads": f["heads"],
                    "kv_heads": kv,
                    "vocab": 8192,
                    "file_size_mb": file_size_mb,
                    "depth": d,
                    "depth_label": depth_label,
                    "tg_ts": res["tg_ts"],
                    "tg_ts_std": res["tg_ts_std"],
                    "pp_ts": res["pp_ts"],
                    "pp_ts_std": res["pp_ts_std"],
                    "retention_pct": retention_pct,
                    "peak_vram_mib": res["peak_vram_mib"],
                    "delta_vram_mib": res["delta_vram_mib"],
                    "gpu_temp_c": res["gpu_temp_c"],
                    "gpu_power_w": res["gpu_power_w"],
                    "gpu_clock_mhz": res["gpu_clock_mhz"],
                }
                all_results.append(row)
                print(f"TG={res['tg_ts']:.1f} +/- {res['tg_ts_std']:.1f} t/s ({retention_pct}% of d0) | "
                      f"Peak VRAM={res['peak_vram_mib']:.0f} MiB (delta {res['delta_vram_mib']:.0f} MiB) [{dt:.1f}s]")
                      
            if gguf_path.exists():
                gguf_path.unlink()

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved all results to {OUT_CSV}")

    # Generate Plots
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Plot 1: Decode Speed (tg t/s) vs Context Depth
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for model_name in df["model"].unique():
        for kv in [2, 1]:
            sub = df[(df["model"] == model_name) & (df["kv_heads"] == kv)].sort_values("depth")
            if not sub.empty:
                label = f"{model_name} ({kv} KV heads)"
                linestyle = "-" if kv == 2 else "--"
                marker = "o" if kv == 2 else "s"
                ax.errorbar(
                    sub["depth"] / 1000, sub["tg_ts"], yerr=sub["tg_ts_std"],
                    label=label, marker=marker, linestyle=linestyle, capsize=4, capthick=1.5
                )
                
    ax.set_xlabel("Context Depth (k tokens)")
    ax.set_ylabel("Decode Speed tg (tokens/sec)")
    ax.set_title("Decode Speed vs Context Depth on RTX 3070 Ti (5 reps with error bars)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Model & KV Heads")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "long_context_speed.png")
    plt.close()

    # Plot 2: Peak VRAM vs Context Depth
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for model_name in df["model"].unique():
        for kv in [2, 1]:
            sub = df[(df["model"] == model_name) & (df["kv_heads"] == kv)].sort_values("depth")
            if not sub.empty:
                label = f"{model_name} ({kv} KV heads)"
                linestyle = "-" if kv == 2 else "--"
                marker = "o" if kv == 2 else "s"
                ax.plot(sub["depth"] / 1000, sub["peak_vram_mib"], label=label, marker=marker, linestyle=linestyle)
                
    ax.axhline(8192, color="red", linestyle=":", label="RTX 3070 Ti VRAM Limit (8,192 MiB)")
    ax.set_xlabel("Context Depth (k tokens)")
    ax.set_ylabel("Total Peak GPU VRAM (MiB)")
    ax.set_title("Peak VRAM Usage vs Context Depth (Batch 1)")
    ax.set_ylim(0, 8500)
    ax.legend(title="Model & KV Heads", loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "long_context_vram.png")
    plt.close()

    print("Plots generated successfully: results/long_context_speed.png, results/long_context_vram.png")

if __name__ == "__main__":
    main()
