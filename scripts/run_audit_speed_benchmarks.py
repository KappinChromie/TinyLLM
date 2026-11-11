import os
import sys
import json
import time
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BIN_PATH = ROOT / "tools" / "llama-bin"

LLAMA_BENCH = BIN_PATH / "llama-bench.exe"
LLAMA_BATCHED_BENCH = BIN_PATH / "llama-batched-bench.exe"

BASE_GGUF_Q8 = ROOT / "models" / "gguf" / "stageC_decayed_q8_0.gguf"
CHAT_GGUF_Q8 = ROOT / "release_chat" / "tinyllm-chat-q8_0.gguf"

def query_gpu_stats():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return {
            "temp_gpu_c": parts[0],
            "clock_core_mhz": parts[1],
            "clock_mem_mhz": parts[2],
            "power_w": parts[3],
            "vram_used_mb": parts[4],
            "vram_total_mb": parts[5]
        }
    except Exception as e:
        return {"error": str(e)}

def run_decode_speed():
    print("="*75)
    print("RUNNING SINGLE-STREAM DECODE SPEED (10 REPS, BASE VS CHAT)")
    print("="*75, flush=True)
    
    models = [
        ("Base Model (Stage C Decayed Q8_0)", BASE_GGUF_Q8),
        ("Chat Model (SFT Final Q8_0)", CHAT_GGUF_Q8),
    ]
    depths = [0, 8192, 32768]
    records = []
    
    for m_label, m_path in models:
        print(f"\nBenchmarking {m_label} across depths {depths}...", flush=True)
        for d in depths:
            # Run 10 repetitions
            cmd = [
                str(LLAMA_BENCH),
                "-m", str(m_path),
                "-ngl", "99",
                "-fa", "1",
                "-n", "256",
                "-p", "0",
                "-d", str(d),
                "-r", "10",
                "-o", "json"
            ]
            gpu_pre = query_gpu_stats()
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            dur = time.time() - t0
            gpu_post = query_gpu_stats()
            
            raw_stdout = res.stdout.strip()
            json_start = raw_stdout.find("[")
            if json_start != -1:
                data = json.loads(raw_stdout[json_start:])
                item = data[0]
                samples = item.get("samples_ts", [])
                if samples:
                    median_s = float(np.median(samples))
                    min_s = float(np.min(samples))
                    max_s = float(np.max(samples))
                else:
                    median_s = item.get("avg_ts", 0.0)
                    min_s = median_s
                    max_s = median_s
                    
                records.append({
                    "model": m_label,
                    "quant": "Q8_0",
                    "depth": d,
                    "median_tok_s": round(median_s, 2),
                    "min_tok_s": round(min_s, 2),
                    "max_tok_s": round(max_s, 2),
                    "std_tok_s": round(item.get("stddev_ts", 0.0), 2),
                    "repetitions": 10,
                    "gpu_temp_c": gpu_post.get("temp_gpu_c"),
                    "gpu_power_w": gpu_post.get("power_w")
                })
                print(f"  Depth {d:5d}: Median={median_s:7.2f} | Min={min_s:7.2f} | Max={max_s:7.2f} | Temp={gpu_post.get('temp_gpu_c')}C | Power={gpu_post.get('power_w')}W")
                
    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "step2_audited_decode_speed_side_by_side.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved audited decode speed table to {out_path}")
    return df

def run_batched_throughput():
    print("\n" + "="*75)
    print("RUNNING BATCHED THROUGHPUT (CONCURRENCY 1, 2, 4, 8; 2K CTX; P512, G128; 10 REPS)")
    print("="*75, flush=True)
    
    models = [
        ("Base Model (Stage C Decayed Q8_0)", BASE_GGUF_Q8),
        ("Chat Model (SFT Final Q8_0)", CHAT_GGUF_Q8),
    ]
    concurrencies = [1, 2, 4, 8]
    npp = 512
    ntg = 128
    
    records = []
    
    for m_label, m_path in models:
        print(f"\nBenchmarking batched throughput for {m_label}...", flush=True)
        for npl in concurrencies:
            req_ctx = 2048 * npl
            # Run 10 repetitions
            gen_speeds = []
            tot_speeds = []
            temps = []
            powers = []
            
            for rep in range(10):
                cmd = [
                    str(LLAMA_BATCHED_BENCH),
                    "-m", str(m_path),
                    "-c", str(req_ctx),
                    "-b", "2048",
                    "-ub", "512",
                    "-ngl", "99",
                    "-fa", "1",
                    "-npp", str(npp),
                    "-ntg", str(ntg),
                    "-npl", str(npl)
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                gpu_now = query_gpu_stats()
                temps.append(gpu_now.get("temp_gpu_c", 0.0))
                powers.append(gpu_now.get("power_w", 0.0))
                
                # Parse markdown table output
                for line in res.stdout.splitlines():
                    if "|" in line:
                        parts = [p.strip() for p in line.split("|")[1:-1]]
                        if len(parts) >= 10 and parts[0].isdigit():
                            try:
                                s_tg = float(parts[7])
                                s_tot = float(parts[9])
                                gen_speeds.append(s_tg)
                                tot_speeds.append(s_tot)
                            except ValueError:
                                pass
                            
            if gen_speeds:
                med_gen = float(np.median(gen_speeds))
                min_gen = float(np.min(gen_speeds))
                max_gen = float(np.max(gen_speeds))
                med_tot = float(np.median(tot_speeds))
            else:
                med_gen, min_gen, max_gen, med_tot = 0.0, 0.0, 0.0, 0.0
                
            records.append({
                "model": m_label,
                "concurrency": npl,
                "prompt_tokens": npp,
                "gen_tokens": ntg,
                "median_generation_tok_s": round(med_gen, 2),
                "min_generation_tok_s": round(min_gen, 2),
                "max_generation_tok_s": round(max_gen, 2),
                "median_total_throughput_tok_s": round(med_tot, 2),
                "repetitions": 10,
                "mean_gpu_temp_c": round(float(np.mean(temps)), 1),
                "mean_gpu_power_w": round(float(np.mean(powers)), 1)
            })
            print(f"  Concurrency {npl:2d}: Median Gen={med_gen:7.2f} | Min={min_gen:7.2f} | Max={max_gen:7.2f} | Tot={med_tot:7.2f} | Temp={np.mean(temps):.1f}C | Power={np.mean(powers):.1f}W")

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "step2_audited_batched_throughput_side_by_side.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved audited batched throughput table to {out_path}")
    return df

if __name__ == "__main__":
    # run_decode_speed()
    run_batched_throughput()
