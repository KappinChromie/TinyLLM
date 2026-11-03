import os
import sys
import shutil
import subprocess
import csv
import time
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SPM_DIR = MODELS_DIR / "spm_placeholders"
TOOLS_DIR = ROOT / "tools"
LLAMA_BENCH = TOOLS_DIR / "llama-bin" / "llama-bench.exe"
CONVERT_SCRIPT = TOOLS_DIR / "llama.cpp-src" / "convert_hf_to_gguf.py"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SWEEP_CSV = RESULTS_DIR / "sweep.csv"

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
    except Exception:
        return {"temp_c": None, "power_w": None, "clock_mhz": None}

def count_params(vocab, d, layers, ffn, heads, kv_heads, tied=True):
    head_dim = d // heads
    attn = d * d + 2 * d * (kv_heads * head_dim) + d * d
    mlp = 3 * d * ffn
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = vocab * d
    total = layers * per_layer + emb * (1 if tied else 2) + d
    return total, total - emb

def round_to_mult(val, mult):
    return max(mult, int(round(val / mult) * mult))

def build_config_list():
    configs = []
    seen = set()

    def add_cfg(l, h, ffn_ratio, kv_mode, v):
        # Determine heads
        # Standard head_dim 64, or 32 for h=128
        head_dim = 64 if h >= 256 else 32
        heads = h // head_dim
        if kv_mode == "1":
            kv_heads = 1
        elif kv_mode == "2":
            kv_heads = min(2, heads)
        elif kv_mode == "full":
            kv_heads = heads
        else:
            kv_heads = min(int(kv_mode), heads)
            
        # ffn dim
        raw_ffn = h * ffn_ratio
        # Round FFN to multiple of 256 if possible, or 64
        mult = 256 if raw_ffn >= 256 else 64
        ffn = round_to_mult(raw_ffn, mult)
        
        key = (l, h, ffn, kv_heads, v)
        if key not in seen:
            seen.add(key)
            configs.append({
                "layers": l,
                "hidden": h,
                "ffn": ffn,
                "heads": heads,
                "kv_heads": kv_heads,
                "vocab": v,
                "ffn_ratio": ffn_ratio,
                "kv_mode": kv_mode,
            })

    # 1. Full 2D Grid: layers x hidden (standard FFN 2.67x, kv=2, vocab=8192)
    for l in [1, 2, 3, 4, 6, 8, 12]:
        for h in [128, 256, 512, 768, 1024]:
            add_cfg(l, h, 2.67, "2", 8192)

    # 2. Vocab sweep across sizes on 3 reference shapes:
    # Small (2L, 256H), Medium (4L, 512H), Large (6L, 768H)
    for v in [2048, 4096, 8192, 16384, 32000, 49152]:
        add_cfg(2, 256, 2.67, "2", v)
        add_cfg(4, 512, 2.67, "2", v)
        add_cfg(6, 768, 2.67, "2", v)

    # 3. FFN ratio sweep: 2x, 2.67x, 4x on 4 reference shapes
    for ffn_r in [2.0, 2.67, 4.0]:
        add_cfg(2, 256, ffn_r, "2", 8192)
        add_cfg(4, 256, ffn_r, "2", 8192)
        add_cfg(4, 512, ffn_r, "2", 8192)
        add_cfg(6, 512, ffn_r, "2", 8192)

    # 4. KV heads sweep: 1, 2, full on 4 reference shapes
    for kv in ["1", "2", "full"]:
        add_cfg(2, 256, 2.67, kv, 8192)
        add_cfg(4, 512, 2.67, kv, 8192)
        add_cfg(6, 768, 2.67, kv, 8192)
        add_cfg(8, 512, 2.67, kv, 8192)

    # 5. Full 2D Grid at vocab=32000 (standard Llama-2 vocab) for candidate shapes
    for l in [1, 2, 3, 4, 6]:
        for h in [256, 512, 768]:
            add_cfg(l, h, 2.67, "2", 32000)

    # 6. Vocab=49152 (SmolLM2 vocab) for candidate shapes
    for l in [1, 2, 3, 4, 6]:
        for h in [256, 512, 768]:
            add_cfg(l, h, 2.67, "2", 49152)

    # 7. Depth vs Width isocost points (~15M - 40M params)
    # 12L x 256H vs 6L x 384H/512H vs 3L x 768H vs 2L x 1024H
    for l, h in [(12, 256), (8, 384 if 384 in [256, 512, 768] else 512), (6, 512), (3, 768), (2, 1024), (1, 1024)]:
        add_cfg(l, h, 2.67, "2", 8192)
        add_cfg(l, h, 2.67, "2", 32000)

    return configs

def benchmark_config(cfg, tmp_dir, gguf_path):
    vocab = cfg["vocab"]
    spm_file = SPM_DIR / f"spm_{vocab}.model"
    if not spm_file.exists():
        raise FileNotFoundError(f"Missing placeholder tokenizer: {spm_file}")

    # Create dummy HF model
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(spm_file, tmp_dir / "tokenizer.model")

    llama_cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=cfg["hidden"],
        intermediate_size=cfg["ffn"],
        num_hidden_layers=cfg["layers"],
        num_attention_heads=cfg["heads"],
        num_key_value_heads=cfg["kv_heads"],
        tie_word_embeddings=True,
        torch_dtype="float16"
    )
    model = LlamaForCausalLM(llama_cfg).half()
    model.save_pretrained(tmp_dir)
    del model
    torch.cuda.empty_cache()

    # Convert to GGUF
    if gguf_path.exists():
        gguf_path.unlink()
    cmd_convert = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(tmp_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16"
    ]
    subprocess.run(cmd_convert, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    file_size_mb = round(gguf_path.stat().st_size / (1024 * 1024), 2)

    # Benchmark with llama-bench: 1 warmup, 3 repetitions
    cmd_bench = [
        str(LLAMA_BENCH),
        "-m", str(gguf_path),
        "-ngl", "99",
        "-fa", "1",
        "-p", "512",
        "-n", "256",
        "-r", "3",
        "-o", "csv"
    ]
    res = subprocess.run(cmd_bench, capture_output=True, text=True, check=True)
    post_gpu = get_gpu_telemetry()

    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("build_commit"):
            header_idx = i
            break
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)

    pp_row = next((r for r in rows if r["n_prompt"] == "512" and r["n_gen"] == "0"), None)
    tg_row = next((r for r in rows if r["n_prompt"] == "0" and r["n_gen"] == "256"), None)

    # Clean up immediately
    shutil.rmtree(tmp_dir)
    if gguf_path.exists():
        gguf_path.unlink()

    tot_p, non_emb_p = count_params(
        vocab, cfg["hidden"], cfg["layers"], cfg["ffn"], cfg["heads"], cfg["kv_heads"], tied=True
    )

    return {
        "layers": cfg["layers"],
        "hidden": cfg["hidden"],
        "ffn": cfg["ffn"],
        "heads": cfg["heads"],
        "kv_heads": cfg["kv_heads"],
        "vocab": vocab,
        "ffn_ratio": cfg["ffn_ratio"],
        "kv_mode": cfg["kv_mode"],
        "total_params_m": round(tot_p / 1e6, 3),
        "non_emb_params_m": round(non_emb_p / 1e6, 3),
        "file_size_mb": file_size_mb,
        "pp_ts": float(pp_row["avg_ts"]) if pp_row else None,
        "pp_ts_std": float(pp_row["stddev_ts"]) if pp_row else None,
        "tg_ts": float(tg_row["avg_ts"]) if tg_row else None,
        "tg_ts_std": float(tg_row["stddev_ts"]) if tg_row else None,
        "gpu_temp_c": post_gpu["temp_c"],
        "gpu_power_w": post_gpu["power_w"],
        "gpu_clock_mhz": post_gpu["clock_mhz"],
    }

def generate_plots(df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Plot 1: tg t/s vs total params
    plt.figure(figsize=(10, 6), dpi=150)
    sc = plt.scatter(df["total_params_m"], df["tg_ts"], c=df["layers"], cmap="viridis", s=50, alpha=0.85, edgecolors="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Layer Count")
    max_tg = df["tg_ts"].max()
    plt.axhline(max_tg * 0.90, color="green", linestyle="--", label="10% Knee Floor (>= 90% peak)")
    plt.axhline(max_tg * 0.75, color="orange", linestyle="--", label="25% Tier (>= 75% peak)")
    plt.axhline(max_tg * 0.50, color="red", linestyle="--", label="50% Tier (>= 50% peak)")
    plt.xlabel("Total Parameters (Millions)")
    plt.ylabel("Decode Speed tg (tokens/sec)")
    plt.title("Decode Tokens/sec vs Total Parameters (RTX 3070 Ti, Batch 1)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tg_vs_params.png")
    plt.close()

    # Plot 2: tg t/s vs layer count
    plt.figure(figsize=(10, 6), dpi=150)
    for h in sorted(df["hidden"].unique()):
        sub = df[(df["hidden"] == h) & (df["vocab"] == 8192) & (df["ffn_ratio"] == 2.67) & (df["kv_mode"] == "2")]
        if not sub.empty:
            sub = sub.sort_values("layers")
            plt.plot(sub["layers"], sub["tg_ts"], marker="o", label=f"Hidden {h}")
    plt.xlabel("Layer Count")
    plt.ylabel("Decode Speed tg (tokens/sec)")
    plt.title("Decode Tokens/sec vs Layer Count by Hidden Size (Vocab=8k, FFN=2.67x)")
    plt.legend(title="Hidden Size")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tg_vs_layers.png")
    plt.close()

    # Plot 3: tg t/s vs vocab size
    plt.figure(figsize=(10, 6), dpi=150)
    for (l, h) in [(2, 256), (4, 512), (6, 768)]:
        sub = df[(df["layers"] == l) & (df["hidden"] == h) & (df["ffn_ratio"] == 2.67) & (df["kv_mode"] == "2")]
        if not sub.empty:
            sub = sub.sort_values("vocab")
            plt.plot(sub["vocab"], sub["tg_ts"], marker="s", label=f"{l}L x {h}H")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Decode Speed tg (tokens/sec)")
    plt.title("Decode Tokens/sec vs Vocabulary Size (Cost of Distillation Head)")
    plt.legend(title="Model Shape")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tg_vs_vocab.png")
    plt.close()

def main():
    configs = build_config_list()
    print(f"Total configurations in sweep: {len(configs)}")

    tmp_dir = ROOT / "tmp_sweep_model"
    gguf_path = ROOT / "tmp_sweep.gguf"

    # Check existing progress if resume
    completed_keys = set()
    rows = []
    if SWEEP_CSV.exists():
        with open(SWEEP_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (int(r["layers"]), int(r["hidden"]), int(r["ffn"]), int(r["kv_heads"]), int(r["vocab"]))
                completed_keys.add(key)
                # convert numeric fields back
                r_parsed = {}
                for k, v in r.items():
                    try:
                        r_parsed[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        r_parsed[k] = v
                rows.append(r_parsed)
        print(f"Resuming: {len(rows)} configs already completed.")

    fieldnames = [
        "layers", "hidden", "ffn", "heads", "kv_heads", "vocab", "ffn_ratio", "kv_mode",
        "total_params_m", "non_emb_params_m", "file_size_mb", "pp_ts", "pp_ts_std", "tg_ts", "tg_ts_std",
        "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz"
    ]

    with open(SWEEP_CSV, "a" if completed_keys else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not completed_keys:
            writer.writeheader()
            f.flush()

        for idx, cfg in enumerate(configs, start=1):
            key = (cfg["layers"], cfg["hidden"], cfg["ffn"], cfg["kv_heads"], cfg["vocab"])
            if key in completed_keys:
                continue

            t0 = time.time()
            data = benchmark_config(cfg, tmp_dir, gguf_path)
            dt = time.time() - t0
            rows.append(data)
            writer.writerow(data)
            f.flush()

            print(f"[{idx}/{len(configs)}] {cfg['layers']}L/{cfg['hidden']}H/{cfg['ffn']}FFN/v{cfg['vocab']}: "
                  f"TG={data['tg_ts']:.1f} t/s, PP={data['pp_ts']:.0f} t/s ({dt:.1f}s) | GPU: {data['gpu_temp_c']}C, {data['gpu_power_w']}W")

    df = pd.DataFrame(rows)
    generate_plots(df)
    print(f"\nAll {len(rows)} benchmarks finished! Saved sweep results and plots to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
