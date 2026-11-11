import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import ctypes
from ctypes import (
    c_void_p, c_int32, c_uint32, c_float, c_size_t, c_bool, c_int8, c_char_p, Structure, POINTER, CFUNCTYPE
)

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
BIN_PATH = ROOT / "tools" / "llama-bin"

os.environ['PATH'] = str(BIN_PATH) + ';' + os.environ['PATH']
os.add_dll_directory(str(BIN_PATH))

# Llama.cpp Ctypes Binding Setup
lib_ggml = ctypes.CDLL(str(BIN_PATH / "ggml.dll"))
lib_llama = ctypes.CDLL(str(BIN_PATH / "llama.dll"))

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
        self.model_path = str(model_path)
        mparams = lib_llama.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        self.model = lib_llama.llama_model_load_from_file(self.model_path.encode('utf-8'), mparams)
        if not self.model:
            raise RuntimeError(f"Failed to load model from {model_path}")
            
        cparams = lib_llama.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_batch
        cparams.n_ubatch = 512
        cparams.flash_attn_type = 1 if flash_attn else 0
        cparams.offload_kqv = True
        cparams.no_perf = True
        self.ctx = lib_llama.llama_init_from_model(self.model, cparams)
        if not self.ctx:
            raise RuntimeError(f"Failed to create context for {model_path}")
            
        self.vocab_size = 8256
        self.n_batch = n_batch
        self.batch = lib_llama.llama_batch_init(n_batch, 0, 1)

    def close(self):
        if self.batch:
            lib_llama.llama_batch_free(self.batch)
            self.batch = None
        if self.ctx:
            lib_llama.llama_free(self.ctx)
            self.ctx = None
        if self.model:
            lib_llama.llama_model_free(self.model)
            self.model = None

    def clear(self):
        mem = lib_llama.llama_get_memory(self.ctx)
        lib_llama.llama_memory_clear(mem, True)

    def score_sequence(self, tokens):
        L = len(tokens)
        self.clear()
        nll = np.zeros(L - 1, dtype=np.float32)
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
                logits_chunk = np.empty((num_logits, self.vocab_size), dtype=np.float32)
                for i in range(num_logits):
                    ptr = lib_llama.llama_get_logits_ith(self.ctx, i)
                    ctypes.memmove(logits_chunk[i].ctypes.data, ptr, self.vocab_size * 4)
                
                t_logits = torch.from_numpy(logits_chunk)
                log_p = F.log_softmax(t_logits, dim=-1)
                t_targets = torch.from_numpy(tokens[start + 1 : start + 1 + num_logits].astype(np.int64))
                target_nll = -log_p.gather(1, t_targets.unsqueeze(1)).squeeze(1).numpy()
                nll[start : start + num_logits] = target_nll
                
        return nll

@torch.no_grad()
def score_pytorch(model, tokens, device="cuda"):
    inp = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
    tgt = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    out = model(inp)
    loss = F.cross_entropy(out.logits[0].float(), tgt, reduction='none')
    return loss.cpu().numpy()

def main():
    print("="*75)
    print("CHAT MODEL MATCHED PARITY AUDIT (GGUF VS PYTORCH)")
    print("="*75)
    
    chat_f16 = ROOT / "release_chat" / "tinyllm-chat-f16.gguf"
    chat_q8 = ROOT / "release_chat" / "tinyllm-chat-q8_0.gguf"
    chat_q4 = ROOT / "release_chat" / "tinyllm-chat-q4_0.gguf"
    
    # 1. Load PyTorch Chat Model
    cfg = LlamaConfig(
        vocab_size=8256, hidden_size=768, intermediate_size=2048,
        num_hidden_layers=6, num_attention_heads=12, num_key_value_heads=1,
        tie_word_embeddings=True, torch_dtype="bfloat16", _attn_implementation="sdpa",
        max_position_embeddings=32768, rope_theta=1000000.0
    )
    pt_model = LlamaForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16)
    ckpt = torch.load(ROOT / "models" / "checkpoints" / "chat_sft_final.pt", map_location="cuda")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    pt_model.load_state_dict(state)
    pt_model.eval()
    
    # Benchmark subsets (exact same tokens as base audit)
    eval_benchmarks = [
        ("val_wikipedia_neutral", DATA_DIR / "val_wikipedia_neutral.bin", 2048, 976),
        ("val_fineweb_edu", DATA_DIR / "val_fineweb_edu.bin", 2048, 976),
        ("val_cosmopedia", DATA_DIR / "val_cosmopedia.bin", 2048, 976),
        ("val_tinystories", DATA_DIR / "val_tinystories.bin", 2048, 976),
        ("val_pg19_32k", DATA_DIR / "val_pg19_long.bin", 32768, 10),
    ]
    
    # PyTorch evaluation
    pt_results = {}
    print("\n--- Evaluating PyTorch Reference ---", flush=True)
    for name, path, seq_len, num_seqs in eval_benchmarks:
        shard = np.memmap(path, dtype=np.uint16, mode="r")
        nlls = []
        for i in range(num_seqs):
            seq = np.array(shard[i * seq_len : (i + 1) * seq_len], dtype=np.int64)
            loss_arr = score_pytorch(pt_model, seq)
            nlls.append(loss_arr.mean())
        mean_ce = float(np.mean(nlls))
        pt_results[name] = mean_ce
        print(f"  PyTorch Chat on {name} ({num_seqs} seqs x {seq_len}): {mean_ce:.4f} CE", flush=True)
        
    del pt_model
    torch.cuda.empty_cache()
    
    # llama.cpp evaluations
    quants = [("F16", chat_f16), ("Q8_0", chat_q8), ("Q4_0", chat_q4)]
    quant_results = {q_name: {} for q_name, _ in quants}
    
    for q_name, q_path in quants:
        print(f"\n--- Evaluating llama.cpp {q_name} ---", flush=True)
        session = LlamaCppSession(q_path, n_ctx=32768, flash_attn=True)
        for name, path, seq_len, num_seqs in eval_benchmarks:
            shard = np.memmap(path, dtype=np.uint16, mode="r")
            nlls = []
            for i in range(num_seqs):
                seq = np.array(shard[i * seq_len : (i + 1) * seq_len], dtype=np.int64)
                loss_arr = session.score_sequence(seq)
                nlls.append(loss_arr.mean())
            mean_ce = float(np.mean(nlls))
            quant_results[q_name][name] = mean_ce
            print(f"  llama.cpp {q_name} on {name}: {mean_ce:.4f} CE", flush=True)
        session.close()

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
            'pytorch_ce': round(pt_ce, 4),
            'llama_f16_ce': round(f16_ce, 4),
            'llama_q8_0_ce': round(q8_ce, 4),
            'llama_q4_0_ce': round(q4_ce, 4),
            'delta_f16_vs_pytorch': round(f16_ce - pt_ce, 4),
            'delta_q8_vs_f16': round(q8_ce - f16_ce, 4),
            'delta_q4_vs_f16': round(q4_ce - f16_ce, 4)
        })
        
    df_quant = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "step2_chat_matched_quantization_table.csv"
    df_quant.to_csv(out_csv, index=False)
    print(f"\nSaved chat matched quantization table to {out_csv}")
    print(df_quant.to_string())

if __name__ == "__main__":
    main()
