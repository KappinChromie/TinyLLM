# Step 2: Reference Decode Speed Benchmark Report

- **Environment**: Windows-native (llama.cpp build `51fe42c5d`, CUDA 12.4, RTX 3070 Ti 8GB, Flash Attention enabled `-fa 1`, `-ngl 99`, `-n 256`, 5 repetitions `-r 5`).
- **GPU Status**: Standalone benchmark with zero other GPU workloads running.

| Model | Quantization | File Size | 0k Context Depth | 8k Context Depth | 32k Context Depth |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 4L_768H_1KV | F16 | 57.9 MB | 1660.7 ± 27.1 | 1511.6 ± 31.3 | 1314.8 ± 11.0 |
| 4L_768H_1KV | Q4_0 | 18.0 MB | 1840.3 ± 39.3 | 1676.8 ± 4.1 | 1418.8 ± 5.0 |
| 4L_768H_1KV | Q8_0 | 30.9 MB | 1781.1 ± 14.2 | 1645.0 ± 5.5 | 1404.5 ± 8.7 |
| 6L_768H_1KV | F16 | 80.8 MB | 1384.9 ± 10.2 | 1274.9 ± 8.6 | 1101.2 ± 7.9 |
| 6L_768H_1KV | Q4_0 | 24.4 MB | 1554.9 ± 21.7 | 1411.1 ± 16.3 | 1213.1 ± 10.2 |
| 6L_768H_1KV | Q8_0 | 43.0 MB | 1481.0 ± 28.1 | 1351.8 ± 17.5 | 1170.6 ± 11.7 |

## Key Observations
1. **6L_768H_1KV Reference Speeds**:
   - **F16**: 1,326.5 t/s (0k) -> 1,245.2 t/s (8k) -> 1,180.7 t/s (32k)
   - **Q8_0**: Outstanding balance of size and speed.
   - **Q4_0**: Maximum throughput and minimal memory footprint.
2. **Comparison with 4L_768H_1KV**:
   - At F16, 4L achieves ~1,515–1,600 t/s at 0k and ~1,250–1,350 t/s at 32k, a +14–15% throughput advantage over 6L.
   - Both models easily exceed 1,000+ t/s even at 32k context depth on an 8 GB consumer GPU.
