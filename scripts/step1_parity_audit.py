import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

import ctypes
from ctypes import (
    c_void_p, c_int32, c_uint32, c_float, c_size_t, c_bool, c_int8, c_char_p, Structure, POINTER, CFUNCTYPE
)

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BIN_PATH = ROOT / "tools" / "llama-bin"

os.environ['PATH'] = str(BIN_PATH) + ';' + os.environ['PATH']
os.add_dll_directory(str(BIN_PATH))

# ---------------------------------------------------------
# Llama.cpp Ctypes Binding Setup
# ---------------------------------------------------------
lib_ggml = ctypes.CDLL(str(BIN_PATH / "ggml.dll"))
lib_llama = ctypes.CDLL(str(BIN_PATH / "llama.dll"))

# Struct definitions matching llama.h
class LlamaModelParams(Structure):
    _fields_ = [
        ('devices', c_void_p), ('tensor_buft_overrides', c_void_p), ('n_gpu_layers', c_int32),
        ('split_mode', c_int32), ('load_mode', c_int32), ('lazy_mode', c_int32), ('main_gpu', c_int32),
        ('tensor_split', c_void_p), ('progress_callback', c_void_p), ('progress_callback_user_data', c_void_p),
        ('kv_overrides', c_void_p), ('vocab_only', c_bool), ('check_tensors', c_bool),
        ('use_extra_bufts', c_bool), ('no_host', c_bool), ('no_alloc', c_bool), ('load_mtp', c_bool)
    ]

class LlamaContextParams(Structure):
    _fields_ = [
        ('n_ctx', c_uint32), ('n_batch', c_uint32), ('n_ubatch', c_uint32), ('n_seq_max', c_uint32),
        ('n_rs_seq', c_uint32), ('n_outputs_max', c_uint32), ('n_outputs_max_per_seq', c_uint32),
        ('n_threads', c_int32), ('n_threads_batch', c_int32), ('ctx_type', c_int32),
        ('rope_scaling_type', c_int32), ('pooling_type', c_int32), ('attention_type', c_int32),
        ('flash_attn_type', c_int32), ('rope_freq_base', c_float), ('rope_freq_scale', c_float),
        ('yarn_ext_factor', c_float), ('yarn_attn_factor', c_float), ('yarn_beta_fast', c_float),
        ('yarn_beta_slow', c_float), ('yarn_orig_ctx', c_uint32), ('defrag_thold', c_float),
        ('cb_eval', c_void_p), ('cb_eval_user_data', c_void_p), ('type_k', c_int32), ('type_v', c_int32),
        ('moe_cache_size', c_size_t), ('abort_callback', c_void_p), ('abort_callback_data', c_void_p),
        ('embeddings', c_bool), ('offload_kqv', c_bool), ('no_perf', c_bool), ('op_offload', c_bool),
        ('swa_full', c_bool), ('kv_unified', c_bool), ('samplers', c_void_p), ('n_samplers', c_size_t),
        ('ctx_other', c_void_p)
    ]

class LlamaBatch(Structure):
    _fields_ = [
        ('n_tokens', c_int32), ('token', POINTER(c_int32)), ('embd', POINTER(c_float)),
        ('pos', POINTER(c_int32)), ('n_seq_id', POINTER(c_int32)), ('seq_id', POINTER(POINTER(c_int32))),
        ('logits', POINTER(c_int8))
    ]

# Function signatures
LOG_CALLBACK = CFUNCTYPE(None, c_int32, c_char_p, c_void_p)
def dummy_log(level, text, user_data):
    pass
c_dummy_log = LOG_CALLBACK(dummy_log)

lib_ggml.ggml_backend_load(str(BIN_PATH / "ggml-cuda.dll").encode('utf-8'))
lib_ggml.ggml_backend_load_all_from_path(str(BIN_PATH).encode('utf-8'))
lib_llama.llama_backend_init()
lib_llama.llama_log_set(c_dummy_log, None)

lib_llama.llama_model_default_params.restype = LlamaModelParams
lib_llama.llama_context_default_params.restype = LlamaContextParams
lib_llama.llama_model_load_from_file.argtypes = [c_char_p, LlamaModelParams]
lib_llama.llama_model_load_from_file.restype = c_void_p
lib_llama.llama_init_from_model.argtypes = [c_void_p, LlamaContextParams]
lib_llama.llama_init_from_model.restype = c_void_p
lib_llama.llama_batch_init.argtypes = [c_int32, c_int32, c_int32]
lib_llama.llama_batch_init.restype = LlamaBatch
lib_llama.llama_batch_free.argtypes = [LlamaBatch]
lib_llama.llama_batch_free.restype = None
lib_llama.llama_decode.argtypes = [c_void_p, LlamaBatch]
lib_llama.llama_decode.restype = c_int32
lib_llama.llama_get_logits_ith.argtypes = [c_void_p, c_int32]
lib_llama.llama_get_logits_ith.restype = POINTER(c_float)
lib_llama.llama_free.argtypes = [c_void_p]
lib_llama.llama_model_free.argtypes = [c_void_p]
lib_llama.llama_memory_clear.argtypes = [c_void_p, c_bool]
lib_llama.llama_get_memory.argtypes = [c_void_p]
lib_llama.llama_get_memory.restype = c_void_p

class LlamaCppSession:
    def __init__(self, model_path, n_ctx=32768, flash_attn=True, n_gpu_layers=99, n_batch=2048):
        self.mp = lib_llama.llama_model_default_params()
        self.mp.n_gpu_layers = n_gpu_layers
        self.model = lib_llama.llama_model_load_from_file(str(model_path).encode('utf-8'), self.mp)
        if not self.model:
            raise RuntimeError(f"Failed to load model from {model_path}")
            
        self.cp = lib_llama.llama_context_default_params()
        self.cp.n_ctx = n_ctx
        self.cp.n_batch = n_batch
        self.cp.n_ubatch = 512
        self.cp.flash_attn_type = 1 if flash_attn else 0
        self.ctx = lib_llama.llama_init_from_model(self.model, self.cp)
        if not self.ctx:
            raise RuntimeError(f"Failed to create context for {model_path}")
            
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.batch = lib_llama.llama_batch_init(n_batch, 0, 1)

    def clear(self):
        mem = lib_llama.llama_get_memory(self.ctx)
        lib_llama.llama_memory_clear(mem, True)

    def close(self):
        lib_llama.llama_batch_free(self.batch)
        lib_llama.llama_free(self.ctx)
        lib_llama.llama_model_free(self.model)

    def score_sequence(self, tokens, return_logprobs=True):
        """
        Takes 1D numpy array of token IDs, length L <= n_ctx.
        Returns:
          nll: numpy array of length L-1 (NLL for predicting tokens[1:] from positions 0..L-2)
          chunk_logprobs: list of torch tensors if return_logprobs else None
        """
        L = len(tokens)
        self.clear()
        
        nll = np.zeros(L - 1, dtype=np.float32)
        chunk_logprobs = [] if return_logprobs else None
        
        num_chunks = (L + self.n_batch - 1) // self.n_batch
        for c in range(num_chunks):
            start = c * self.n_batch
            end = min(L, start + self.n_batch)
            chunk_len = end - start
            
            self.batch.n_tokens = chunk_len
            for i in range(chunk_len):
                pos = start + i
                self.batch.token[i] = int(tokens[pos])
                self.batch.pos[i] = pos
                self.batch.n_seq_id[i] = 1
                self.batch.seq_id[i][0] = 0
                self.batch.logits[i] = 1 if pos < L - 1 else 0
                
            ret = lib_llama.llama_decode(self.ctx, self.batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode failed with error code {ret} at chunk {c}")
                
            num_logits = chunk_len if end < L else chunk_len - 1
            if num_logits > 0:
                logits_chunk = np.empty((num_logits, 8192), dtype=np.float32)
                for i in range(num_logits):
                    ptr = lib_llama.llama_get_logits_ith(self.ctx, i)
                    ctypes.memmove(logits_chunk[i].ctypes.data, ptr, 8192 * 4)
                
                # Compute log_softmax
                t_logits = torch.from_numpy(logits_chunk)
                log_p = F.log_softmax(t_logits, dim=-1)
                
                # Targets for this chunk
                t_targets = torch.from_numpy(tokens[start + 1 : start + 1 + num_logits].astype(np.int64))
                target_nll = -log_p.gather(1, t_targets.unsqueeze(1)).squeeze(1).numpy()
                
                nll[start : start + num_logits] = target_nll
                if return_logprobs:
                    chunk_logprobs.append(log_p)
                
        return nll, chunk_logprobs

print("Ctypes LlamaCppSession initialized successfully.")
# ---------------------------------------------------------
# PyTorch Scoring Helper
# ---------------------------------------------------------
def score_pytorch(model, tokens, dtype=torch.float32, chunk_size=2048):
    """
    Evaluates tokens (length L) in PyTorch.
    To avoid excessive VRAM allocation for large windows,
    we run the forward pass and extract target NLL and chunk logprobs.
    """
    L = len(tokens)
    device = next(model.parameters()).device
    x = torch.from_numpy(tokens[:L-1].astype(np.int64)).unsqueeze(0).to(device)
    y = torch.from_numpy(tokens[1:L].astype(np.int64)).to(device)
    
    nll = np.zeros(L - 1, dtype=np.float32)
    chunk_logprobs = []
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else torch.no_grad():
            out = model(x)
            logits = out.logits[0] # (L-1, 8192)
            
            # Process in chunks of chunk_size to compute log_softmax and target NLL
            num_chunks = (L - 1 + chunk_size - 1) // chunk_size
            for c in range(num_chunks):
                start = c * chunk_size
                end = min(L - 1, start + chunk_size)
                l_chunk = logits[start:end].float() # compute softmax in float32
                log_p = F.log_softmax(l_chunk, dim=-1)
                t_targets = y[start:end]
                target_nll = -log_p.gather(1, t_targets.unsqueeze(1)).squeeze(1).cpu().numpy()
                nll[start:end] = target_nll
                chunk_logprobs.append(log_p.cpu())
                
    return nll, chunk_logprobs

def compute_kl_divergence(ref_logprobs, cand_logprobs):
    """
    Computes per-position KL divergence:
    KL(P_ref || P_cand) = sum(P_ref * (log_P_ref - log_P_cand))
    Takes lists of chunk logprobs (PyTorch tensors).
    """
    kl_vals = []
    for lp_ref, lp_cand in zip(ref_logprobs, cand_logprobs):
        p_ref = torch.exp(lp_ref)
        kl = (p_ref * (lp_ref - lp_cand)).sum(dim=-1).numpy()
        kl_vals.append(kl)
    return np.concatenate(kl_vals, axis=0)

# ---------------------------------------------------------
# Step 1.1 & 1.2: 20 Held-out PG-19 Windows of 32k Tokens
# ---------------------------------------------------------
def run_32k_parity_sweep():
    print("\n=======================================================", flush=True)
    print("STEP 1: 32k Context Per-Position Parity Audit (20 Windows)", flush=True)
    print("=======================================================", flush=True)
    
    val_pg19_path = DATA_DIR / "val_pg19_long.bin"
    val_pg19 = np.memmap(val_pg19_path, dtype=np.uint16, mode="r")
    total_tokens = len(val_pg19)
    print(f"Loaded PG-19 Val: {total_tokens:,} tokens total.", flush=True)
    
    # Select 20 evenly spaced windows of 32,768 tokens
    W_LEN = 32768
    num_windows = 20
    max_start = total_tokens - W_LEN - 1
    step_size = max_start // num_windows
    rng = np.random.RandomState(42)
    window_starts = [i * step_size + rng.randint(0, min(1000, step_size)) for i in range(num_windows)]
    print(f"Selected {num_windows} fixed evaluation windows (length {W_LEN:,}).", flush=True)
    
    # Buckets
    buckets = [
        ("0-1k", 0, 1024),
        ("1-2k", 1024, 2048),
        ("2-4k", 2048, 4096),
        ("4-8k", 4096, 8192),
        ("8-16k", 8192, 16384),
        ("16-32k", 16384, 32767)
    ]
    
    # 1. Load PyTorch model
    hf_path = ROOT / "models" / "hf_stageC_final"
    print("Loading PyTorch model...", flush=True)
    pt_model = LlamaForCausalLM.from_pretrained(hf_path, torch_dtype=torch.float32).cuda().eval()
    
    # 2. Setup candidate engines
    f16_gguf = ROOT / "models" / "gguf" / "stageC_decayed_f16.gguf"
    q8_gguf = ROOT / "models" / "gguf" / "stageC_decayed_q8_0.gguf"
    
    print("Initializing llama.cpp sessions...", flush=True)
    session_f16_fa = LlamaCppSession(f16_gguf, n_ctx=W_LEN, flash_attn=True)
    session_f16_nofa = LlamaCppSession(f16_gguf, n_ctx=W_LEN, flash_attn=False)
    session_q8_fa = LlamaCppSession(q8_gguf, n_ctx=W_LEN, flash_attn=True)
    
    candidates = [
        ("PyTorch_FP16", None),
        ("llama_F16_FA_on", session_f16_fa),
        ("llama_F16_FA_off", session_f16_nofa),
        ("llama_Q8_0_FA_on", session_q8_fa)
    ]
    
    # Accumulators per candidate and per bucket:
    # metrics[cand][bucket] = {'diffs': [], 'abs_diffs': [], 'kls': []}
    metrics = {cand_name: {b[0]: {'diffs': [], 'abs_diffs': [], 'kls': []} for b in buckets} for cand_name, _ in candidates}
    overall_losses = {'PyTorch_FP32': []}
    for cand_name, _ in candidates:
        overall_losses[cand_name] = []
        
    for w_idx, start_pos in enumerate(window_starts):
        raw_tokens = np.array(val_pg19[start_pos : start_pos + W_LEN], dtype=np.int64)
        print(f"Scoring Window {w_idx+1:02d}/{num_windows} (Tokens {start_pos:,} to {start_pos+W_LEN:,})...", flush=True)
        
        # 1. PyTorch FP32 Reference
        ref_nll, ref_logprobs = score_pytorch(pt_model, raw_tokens, dtype=torch.float32)
        overall_losses['PyTorch_FP32'].append(float(ref_nll.mean()))
        
        # 2. PyTorch FP16
        fp16_nll, fp16_logprobs = score_pytorch(pt_model, raw_tokens, dtype=torch.float16)
        overall_losses['PyTorch_FP16'].append(float(fp16_nll.mean()))
        fp16_kl = compute_kl_divergence(ref_logprobs, fp16_logprobs)
        del fp16_logprobs
        
        diff_fp16 = fp16_nll - ref_nll
        abs_diff_fp16 = np.abs(diff_fp16)
        for b_name, b_start, b_end in buckets:
            metrics['PyTorch_FP16'][b_name]['diffs'].extend(diff_fp16[b_start:b_end])
            metrics['PyTorch_FP16'][b_name]['abs_diffs'].extend(abs_diff_fp16[b_start:b_end])
            metrics['PyTorch_FP16'][b_name]['kls'].extend(fp16_kl[b_start:b_end])
            
        # 3. llama.cpp candidates
        for cand_name, session in candidates[1:]:
            cand_nll, cand_logprobs = session.score_sequence(raw_tokens)
            overall_losses[cand_name].append(float(cand_nll.mean()))
            cand_kl = compute_kl_divergence(ref_logprobs, cand_logprobs)
            del cand_logprobs
            diff = cand_nll - ref_nll
            abs_diff = np.abs(diff)
            for b_name, b_start, b_end in buckets:
                metrics[cand_name][b_name]['diffs'].extend(diff[b_start:b_end])
                metrics[cand_name][b_name]['abs_diffs'].extend(abs_diff[b_start:b_end])
                metrics[cand_name][b_name]['kls'].extend(cand_kl[b_start:b_end])
        del ref_logprobs
                
    # Close sessions
    session_f16_fa.close()
    session_f16_nofa.close()
    session_q8_fa.close()
    del pt_model
    torch.cuda.empty_cache()
    
    # Compile bucket summary
    records = []
    for cand_name, _ in candidates:
        for b_name, b_start, b_end in buckets:
            diffs = np.array(metrics[cand_name][b_name]['diffs'])
            abs_diffs = np.array(metrics[cand_name][b_name]['abs_diffs'])
            kls = np.array(metrics[cand_name][b_name]['kls'])
            records.append({
                'candidate': cand_name,
                'position_bucket': b_name,
                'start_pos': b_start,
                'end_pos': b_end,
                'num_tokens': len(diffs),
                'mean_diff_ce': float(diffs.mean()),
                'mean_abs_diff_ce': float(abs_diffs.mean()),
                'kl_divergence': float(kls.mean())
            })
            
    df_buckets = pd.DataFrame(records)
    df_buckets.to_csv(RESULTS_DIR / "step1_parity_32k_buckets.csv", index=False)
    print("\nSaved 32k parity bucket results to results/step1_parity_32k_buckets.csv", flush=True)
    
    # Overall summary
    summary_records = []
    ref_loss = np.mean(overall_losses['PyTorch_FP32'])
    for name, losses in overall_losses.items():
        mean_l = float(np.mean(losses))
        std_l = float(np.std(losses) / np.sqrt(len(losses)))
        delta_vs_ref = mean_l - ref_loss
        summary_records.append({
            'runtime_model': name,
            'mean_loss_ce': mean_l,
            'se_loss_ce': std_l,
            'delta_vs_pytorch_fp32': delta_vs_ref
        })
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(RESULTS_DIR / "step1_parity_summary.csv", index=False)
    print("Saved 32k parity summary to results/step1_parity_summary.csv", flush=True)
    print(df_summary.to_string())
    
    return df_buckets, df_summary

# ---------------------------------------------------------
# Step 1.4: Reconciled Quantization Quality Table (Exact Matched Scoring)
# ---------------------------------------------------------
def run_matched_quantization_table():
    print("\n=======================================================", flush=True)
    print("STEP 1.4: Reconciled Matched-Scoring Quantization Table", flush=True)
    print("=======================================================", flush=True)
    
    hf_path = ROOT / "models" / "hf_stageC_final"
    f16_gguf = ROOT / "models" / "gguf" / "stageC_decayed_f16.gguf"
    q8_gguf = ROOT / "models" / "gguf" / "stageC_decayed_q8_0.gguf"
    q4_gguf = ROOT / "models" / "gguf" / "stageC_decayed_q4_0.gguf"
    
    eval_benchmarks = [
        ("val_wikipedia_neutral", DATA_DIR / "val_wikipedia_neutral.bin", 2048, 976),
        ("val_fineweb_edu", DATA_DIR / "val_fineweb_edu.bin", 2048, 976),
        ("val_cosmopedia", DATA_DIR / "val_cosmopedia.bin", 2048, 976),
        ("val_tinystories", DATA_DIR / "val_tinystories.bin", 2048, 976),
        ("val_pg19_32k", DATA_DIR / "val_pg19_long.bin", 32768, 10) # 10 full 32k windows
    ]
    
    # 1. PyTorch FP32 Baseline
    print("Evaluating PyTorch FP32 Baseline...", flush=True)
    pt_model = LlamaForCausalLM.from_pretrained(hf_path, torch_dtype=torch.float32).cuda().eval()
    pt_results = {}
    for name, path, seq_len, num_seqs in eval_benchmarks:
        shard = np.memmap(path, dtype=np.uint16, mode="r")
        total_eval_tokens = seq_len * num_seqs
        assert len(shard) >= total_eval_tokens
        
        if seq_len == 2048:
            bs = 16
            total_loss = 0.0
            total_tokens = 0
            for b_idx in range(0, num_seqs, bs):
                cur_bs = min(bs, num_seqs - b_idx)
                batch_tokens = shard[b_idx * seq_len : (b_idx + cur_bs) * seq_len].reshape(cur_bs, seq_len).astype(np.int64)
                x = torch.from_numpy(batch_tokens[:, :-1]).cuda()
                y = torch.from_numpy(batch_tokens[:, 1:]).cuda()
                with torch.no_grad():
                    out = pt_model(x)
                    loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1), reduction="sum").item()
                    total_loss += loss
                    total_tokens += cur_bs * (seq_len - 1)
            mean_ce = total_loss / total_tokens
        else:
            nlls = []
            for i in range(num_seqs):
                seq = np.array(shard[i * seq_len : (i + 1) * seq_len], dtype=np.int64)
                nll, _ = score_pytorch(pt_model, seq, dtype=torch.float32)
                nlls.append(nll.mean())
            mean_ce = float(np.mean(nlls))
            
        pt_results[name] = mean_ce
        print(f"  PyTorch FP32 on {name} ({num_seqs} seqs x {seq_len}): {mean_ce:.4f} CE", flush=True)
    del pt_model
    torch.cuda.empty_cache()
    
    # 2. llama.cpp F16, Q8_0, Q4_0
    quants = [("F16", f16_gguf), ("Q8_0", q8_gguf), ("Q4_0", q4_gguf)]
    quant_results = {q_name: {} for q_name, _ in quants}
    
    for q_name, q_path in quants:
        print(f"Evaluating llama.cpp {q_name}...", flush=True)
        session = LlamaCppSession(q_path, n_ctx=32768, flash_attn=True)
        for name, path, seq_len, num_seqs in eval_benchmarks:
            shard = np.memmap(path, dtype=np.uint16, mode="r")
            nlls = []
            for i in range(num_seqs):
                seq = np.array(shard[i * seq_len : (i + 1) * seq_len], dtype=np.int64)
                nll, _ = session.score_sequence(seq, return_logprobs=False)
                nlls.append(nll.mean())
            mean_ce = float(np.mean(nlls))
            quant_results[q_name][name] = mean_ce
            print(f"  llama.cpp {q_name} on {name}: {mean_ce:.4f} CE", flush=True)
        session.close()
        
    # Build final matched quantization table
    records = []
    for name, _, seq_len, num_seqs in eval_benchmarks:
        pt_ce = pt_results[name]
        f16_ce = quant_results['F16'][name]
        q8_ce = quant_results['Q8_0'][name]
        q4_ce = quant_results['Q4_0'][name]
        
        records.append({
            'dataset': name,
            'context_len': seq_len,
            'num_sequences': num_seqs,
            'eval_tokens': seq_len * num_seqs,
            'pytorch_fp32_ce': round(pt_ce, 4),
            'llama_f16_ce': round(f16_ce, 4),
            'llama_q8_0_ce': round(q8_ce, 4),
            'llama_q4_0_ce': round(q4_ce, 4),
            'delta_f16_vs_pytorch': round(f16_ce - pt_ce, 4),
            'delta_q8_vs_f16': round(q8_ce - f16_ce, 4),
            'delta_q4_vs_f16': round(q4_ce - f16_ce, 4)
        })
        
    df_quant = pd.DataFrame(records)
    df_quant.to_csv(RESULTS_DIR / "step1_matched_quantization_table.csv", index=False)
    print("\nSaved matched quantization table to results/step1_matched_quantization_table.csv", flush=True)
    print(df_quant.to_string())
    return df_quant

if __name__ == "__main__":
    df_buckets, df_summary = run_32k_parity_sweep()
    df_quant = run_matched_quantization_table()
    
    # Check STOP RULE
    # STOP RULE: if the matched-scoring llama.cpp vs PyTorch gap is more than 0.01 CE on average,
    # or grows with position, STOP after this step, report the numbers, and wait for me.
    f16_summary_delta = df_summary[df_summary['runtime_model'] == 'llama_F16_FA_on']['delta_vs_pytorch_fp32'].iloc[0]
    print(f"\n>>> STOP RULE CHECK: Average F16 vs PyTorch FP32 Delta = {f16_summary_delta:+.6f} CE (Threshold: +/-0.01 CE)")
    
    # Check if delta grows with position
    f16_buckets = df_buckets[df_buckets['candidate'] == 'llama_F16_FA_on']
    bucket_diffs = f16_buckets['mean_diff_ce'].tolist()
    print(f">>> Position bucket deltas: {bucket_diffs}")
    
    if abs(f16_summary_delta) > 0.01:
        print("!!! STOP RULE TRIGGERED: Mean delta exceeds 0.01 CE !!!")
        sys.exit(101)
    print(">>> STOP RULE PASSED: Parity is verified within tolerance!")
