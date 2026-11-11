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

GGUF_F16 = ROOT / "models" / "gguf" / "stageC_decayed_f16.gguf"
GGUF_Q8 = ROOT / "models" / "gguf" / "stageC_decayed_q8_0.gguf"
GGUF_Q4 = ROOT / "models" / "gguf" / "stageC_decayed_q4_0.gguf"

def query_gpu_stats():
    """Queries GPU temperature, clock, power, and VRAM using nvidia-smi."""
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

def run_single_stream_benchmark():
    print("\n=======================================================", flush=True)
    print("STEP 4.1: Single-Stream Decode Benchmark (10 Repetitions with Warmup)", flush=True)
    print("=======================================================", flush=True)
    
    models = [("F16", GGUF_F16), ("Q8_0", GGUF_Q8), ("Q4_0", GGUF_Q4)]
    depths = [0, 8192, 16384, 32768]
    records = []
    
    for q_label, m_path in models:
        print(f"\nBenchmarking {q_label} ({m_path.name}) across depths {depths}...", flush=True)
        gpu_pre = query_gpu_stats()
        print(f"  Pre-run GPU: Temp {gpu_pre.get('temp_gpu_c')}C | VRAM {gpu_pre.get('vram_used_mb')} MB | Core {gpu_pre.get('clock_core_mhz')} MHz")
        
        d_str = ",".join(str(d) for d in depths)
        cmd = [
            str(LLAMA_BENCH),
            "-m", str(m_path),
            "-ngl", "99",
            "-fa", "1",
            "-n", "256",
            "-p", "0",
            "-d", d_str,
            "-r", "10",
            "-o", "json"
        ]
        
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur = time.time() - t0
        gpu_post = query_gpu_stats()
        print(f"  Post-run GPU: Temp {gpu_post.get('temp_gpu_c')}C | Power {gpu_post.get('power_w')}W | VRAM {gpu_post.get('vram_used_mb')} MB (took {dur:.1f}s)")
        
        # Clean stdout: filter out any backend logs before the json array '['
        raw_stdout = res.stdout.strip()
        json_start = raw_stdout.find("[")
        if json_start != -1:
            json_text = raw_stdout[json_start:]
            try:
                data = json.loads(json_text)
                for item in data:
                    d_val = item.get("n_depth", 0)
                    gen_speed = item.get("avg_ts", 0.0)
                    gen_err = item.get("stddev_ts", 0.0)
                    records.append({
                        "quant": q_label,
                        "context_depth": d_val,
                        "prompt_tokens": 0,
                        "gen_tokens": 256,
                        "decode_tok_s": round(gen_speed, 2),
                        "decode_err_tok_s": round(gen_err, 2),
                        "repetitions": 10,
                        "gpu_temp_c": gpu_post.get("temp_gpu_c"),
                        "gpu_power_w": gpu_post.get("power_w")
                    })
                    print(f"    Depth {d_val:5d}: {gen_speed:7.2f} +/- {gen_err:5.2f} tok/s")
            except Exception as e:
                print(f"Error parsing llama-bench json: {e}")
                print("Raw json_text:\n", json_text[:500])
        else:
            print("Could not find json array in llama-bench output:\n", raw_stdout[:500])
            
    df_single = pd.DataFrame(records)
    df_single.to_csv(RESULTS_DIR / "step4_single_stream_decode_speed.csv", index=False)
    print("\nSaved single-stream decode speed results to results/step4_single_stream_decode_speed.csv")
    print(df_single.to_string())
    return df_single

def run_batched_throughput_benchmark():
    print("\n=======================================================", flush=True)
    print("STEP 4.2: Batched Throughput Benchmark (llama-batched-bench)", flush=True)
    print("=======================================================", flush=True)
    
    # We benchmark Q8_0, Q4_0, and F16
    models = [("Q8_0", GGUF_Q8), ("Q4_0", GGUF_Q4), ("F16", GGUF_F16)]
    
    # Configurations required:
    # 1. Short prompt (128 tok) + 256 gen: concurrency 1, 8, 32, 64, 128, 256
    # 2. 2k context per seq (2048 prompt + 256 gen): concurrency 1, 8, 32, 64
    # 3. 8k context per seq (8192 prompt + 256 gen): concurrency 1, 8, 32, 64
    # 4. 32k context per seq (32768 prompt + 256 gen): concurrency 8, 16
    
    test_configs = [
        ("short_128p", 128, 256, [1, 8, 32, 64, 128, 256]),
        ("context_2k", 2048, 256, [1, 8, 32, 64]),
        ("context_8k", 8192, 256, [1, 8, 32, 64]),
        ("context_32k", 32768, 256, [1, 4, 8, 16])
    ]
    
    batched_records = []
    
    for q_label, m_path in models:
        print(f"\n--- Running Batched Benchmarks for {q_label} ({m_path.name}) ---", flush=True)
        for suite_name, npp, ntg, npl_list in test_configs:
            for npl in npl_list:
                req_ctx = (npp + ntg) * npl
                # Theoretical KV cache: 1 KV head * 64 head_dim * 6 layers * 2 (K & V) * 2 bytes (F16) = 1536 bytes/token = 1.5 KB/tok
                expected_kv_mb = (req_ctx * 1536) / (1024 * 1024)
                
                # Check VRAM ceiling (7.0 GB usable)
                if expected_kv_mb > 6500:
                    print(f"  Skipping {suite_name} npl={npl} (Expected KV {expected_kv_mb:.1f} MB exceeds VRAM ceiling)")
                    batched_records.append({
                        "quant": q_label,
                        "suite": suite_name,
                        "prompt_tokens": npp,
                        "gen_tokens": ntg,
                        "concurrency": npl,
                        "total_ctx_tokens": req_ctx,
                        "expected_kv_mb": round(expected_kv_mb, 1),
                        "actual_vram_used_mb": None,
                        "prompt_tok_s": None,
                        "generation_tok_s": None,
                        "aggregate_tok_s": None,
                        "per_stream_tok_s": None,
                        "status": "OOM Prevention"
                    })
                    continue
                    
                cmd = [
                    str(LLAMA_BATCHED_BENCH),
                    "-m", str(m_path),
                    "-c", str(req_ctx),
                    "-b", str(min(2048, req_ctx)),
                    "-ub", "512",
                    "-ngl", "99",
                    "-fa", "1",
                    "-npp", str(npp),
                    "-ntg", str(ntg),
                    "-npl", str(npl),
                    "--output-format", "jsonl"
                ]
                
                gpu_before = query_gpu_stats()
                t0 = time.time()
                res = subprocess.run(cmd, capture_output=True, text=True)
                dur = time.time() - t0
                gpu_after = query_gpu_stats()
                
                prompt_ts = None
                gen_ts = None
                total_ts = None
                
                for line in res.stdout.strip().splitlines():
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            obj = json.loads(line)
                            prompt_ts = obj.get("speed_pp")
                            gen_ts = obj.get("speed_tg")
                            total_ts = obj.get("speed")
                        except Exception:
                            pass
                            
                per_stream = (gen_ts / npl) if (gen_ts and npl > 0) else None
                status = "SUCCESS" if gen_ts is not None else "ERROR"
                vram_used = gpu_after.get("vram_used_mb")
                temp_c = gpu_after.get("temp_gpu_c")
                power_w = gpu_after.get("power_w")
                
                batched_records.append({
                    "quant": q_label,
                    "suite": suite_name,
                    "prompt_tokens": npp,
                    "gen_tokens": ntg,
                    "concurrency": npl,
                    "total_ctx_tokens": req_ctx,
                    "expected_kv_mb": round(expected_kv_mb, 1),
                    "actual_vram_used_mb": vram_used,
                    "prompt_tok_s": round(prompt_ts, 2) if prompt_ts else None,
                    "generation_tok_s": round(gen_ts, 2) if gen_ts else None,
                    "aggregate_tok_s": round(gen_ts, 2) if gen_ts else None,
                    "per_stream_tok_s": round(per_stream, 2) if per_stream else None,
                    "temp_gpu_c": temp_c,
                    "power_w": power_w,
                    "status": status
                })
                
                p_str = f"{prompt_ts:9.2f}" if prompt_ts else "   N/A   "
                g_str = f"{gen_ts:9.2f}" if gen_ts else "   N/A   "
                ps_str = f"{per_stream:7.2f}" if per_stream else "  N/A  "
                print(f"  [{q_label:4s} | {suite_name:11s} | Conc: {npl:3d}] Gen: {g_str} agg tok/s ({ps_str} per stream) | Prompt: {p_str} t/s | VRAM: {vram_used} MB (Exp KV: {expected_kv_mb:5.1f} MB) | Temp: {temp_c}C")
                
    df_batched = pd.DataFrame(batched_records)
    df_batched.to_csv(RESULTS_DIR / "step4_batched_throughput.csv", index=False)
    print("\nSaved batched throughput results to results/step4_batched_throughput.csv")
    return df_batched

if __name__ == "__main__":
    df_single = run_single_stream_benchmark()
    df_batched = run_batched_throughput_benchmark()
    print("\nAll Step 4 benchmarks finished successfully!")
