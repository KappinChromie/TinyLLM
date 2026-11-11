# TinyLLM-Chat (42.28M Parameters)

TinyLLM-Chat is a compact 42.28M parameter chat model fine-tuned from TinyLLM Stage C with long-context support up to 32,768 tokens, trained and benchmarked entirely on a single **NVIDIA GeForce RTX 3070 Ti Desktop GPU (8 GB VRAM, 290 W TGP)**.

> [!WARNING]
> **Capabilities and Limitations**: TinyLLM-Chat is an ultra-lightweight (42M parameter) research model. While it reliably follows conversational turn-taking, maintains polite assistant behavior, and achieves 76.2% greedy / 77.8% sampling `<|end|>` emission, its world knowledge and reasoning capabilities are inherently constrained by parameter scale. It frequently produces factual errors or simple summaries on complex questions. It should not be used in production without extensive task-specific evaluation.

---

## Model Architecture
- **Architecture**: Llama-compatible Transformer Decoder (RMSNorm, RoPE, SwiGLU)
- **Parameters**: 42,283,776 (42.28M)
- **Layers**: 6
- **Hidden Dimension**: 768
- **Intermediate Size**: 2,048
- **Attention Heads**: 12 (Head Dim: 64)
- **KV Heads**: 1 (Grouped-Query Attention 12:1)
- **Vocabulary**: 8,256 tokens (custom SentencePiece + ChatML special tokens)
- **Max Context Length**: 32,768 tokens (RoPE Base $\theta = 1,000,000$)

---

## Quantization Guidance & Speed Benchmarks

Benchmarked on **NVIDIA GeForce RTX 3070 Ti (8 GB VRAM, 290 W limit)** using `llama.cpp` (CUDA 12.4, FlashAttention enabled, 10 repetitions per condition):

### Single-Stream Generation Speed (`llama-bench`, 10 Reps)
*Source: `results/step2_audited_decode_speed_side_by_side.csv`*

| Quantization | File Size | Decode Speed (0k Depth) | Decode Speed (8k Depth) | Decode Speed (32k Depth) | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Q8_0** | 45.2 MB | **1,574.23 tok/s** (min 1,520, max 1,598) | **1,427.61 tok/s** (min 1,410, max 1,443) | **1,232.99 tok/s** (min 1,220, max 1,241) | **Recommended Default**: Negligible $+0.0011$ CE quantization loss vs F16 with peak single-stream speed. |
| **F16** | 84.9 MB | 1,433.40 tok/s | 1,360.10 tok/s | 1,159.90 tok/s | Full unquantized precision; best for multi-stream batched serving. |
| **Q4_0** | 25.7 MB | 1,624.70 tok/s | 1,481.40 tok/s | 1,293.60 tok/s | Edge & memory-constrained environments (<30 MB footprint, $+0.0637$ CE loss penalty). |

### Batched Serving Throughput (`llama-batched-bench`, Q8_0, 2k Context, Prompt 512, Gen 128, 10 Reps)
*Source: `results/step2_audited_batched_throughput_side_by_side.csv`*

| Concurrency | Median Generation tok/s | Min Generation tok/s | Max Generation tok/s | Median Total Throughput tok/s | GPU Temp / Power |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Stream** | 1,444.57 | 1,427.97 | 1,454.28 | 6,422.12 | 56.0°C / 74.5 W |
| **2 Streams** | 2,583.42 | 2,536.86 | 2,606.08 | 11,314.65 | 56.0°C / 75.5 W |
| **4 Streams** | 4,406.67 | 4,360.90 | 4,421.49 | 18,473.25 | 56.6°C / 76.1 W |
| **8 Streams** | **6,788.75** | 6,491.16 | 6,859.96 | **27,278.08** | 57.9°C / 85.4 W |

---

## Chat Prompt Format (ChatML)

TinyLLM-Chat uses the standard ChatML formatting:

```
<|system|>
You are a helpful assistant.<|end|>
<|user|>
Hello!<|end|>
<|assistant|>
Hello! How can I help you today?<|end|>
```

### Usage with llama.cpp
```bash
# Interactive Conversation Mode
llama-cli.exe -m release_chat/tinyllm-chat-q8_0.gguf -cnv -ngl 99

# OpenAI-Compatible Server
llama-server.exe -m release_chat/tinyllm-chat-q8_0.gguf --port 8080 -ngl 99 -c 4096
```

---

## Dataset Provenance & Licenses

| Dataset | Split / Tokens Used | License | Generator / Annotator Model | Generator License Terms & Restrictions | Downstream Redistribution |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **smol-smoltalk** | SFT train (114,008 convs, 56.18M tok) | Apache 2.0 | Meta-Llama-3.1-70B-Instruct | Llama 3.1 Community License: allows commercial use under 700M monthly active users; requires attribution. | Permissive (Apache 2.0) |
| **FineWeb-Edu** | Pretrain & Replay (60% short, 25% long) | ODC-By 1.0 | Snowflake-Arctic-Embed-M + Llama-3-70B-Instruct classifier | Annotation / scoring filter only. Model weights trained on filtered web text. | Attribution required under ODC-By |
| **Cosmopedia v2** | Pretrain & Replay (30% short) | ODC-By 1.0 | Mixtral-8x7B-Instruct-v0.1 | Apache 2.0 (permissive, commercial use allowed). | Attribution required under ODC-By |
| **TinyStories** | Pretrain & Replay (10% short) | CDLA-Sharing-1.0 | GPT-3.5 and GPT-4 | OpenAI Business Terms Sec. 2(c)(iii): Output cannot be used to develop models that compete with OpenAI. CDLA-Sharing requires downstream modifications to be shared. | Share-alike clause; commercial competition restriction |
| **Wikipedia** | Long Pretrain & Replay (25% long) | CC BY-SA 3.0 / 4.0 & GFDL | Human authored | CC BY-SA copyleft on text. Unsettled legal question regarding model weights trained on CC BY-SA. | Attribution & ShareAlike |
| **PG-19** | Long Pretrain & Replay (50% long) | Public Domain | Historical human authors | Pre-1919 Project Gutenberg literature. | Unrestricted Public Domain |

### Model License
`[LICENSE PLACEHOLDER - TO BE SELECTED BY USER]`
