# Step 2: 3-Hour Head-to-Head Comparison Report: 4L_768H_1KV vs 6L_768H_1KV

## Executive Summary

- **Task**: Definitive 3-hour head-to-head comparison between **4L_768H_1KV** (30.28M params) and **6L_768H_1KV** (42.28M params) on the Stage A mix.
- **Protocol**: 2,048 context, batch size 32 sequences (65,536 tokens/step), peak LR = 1e-3, AdamW, flat WSD schedule.
- **Main Run Time**: Exactly 3.00 hours (10,800s) per model.
- **Shared Token Milestones**: Full checkpoints saved and evaluated at 250M, 500M, and 750M tokens, plus 3-hour wall-clock end.
- **Equal-Token Decayed Comparisons**: 381-step (~25M tokens) cosine decay to 0.1x LR branched from every token milestone and from the 3-hour mark without eating into the main 3-hour clock.
- **Empirical Noise Floors**:
  - Validation spread across 2 random seeds: **$\Delta_{val} = 0.0076$**
  - Training spread across 3 random seeds: **$\Delta_{train} = 0.0254$**

## Key Findings

1. **Equal Token Volume (250M, 500M, 750M Tokens)**:
   - At 250M tokens: 6L leads by **-0.1191** decayed CE (**15.7x** validation noise floor).
   - At 500M tokens: 6L leads by **-0.1290** decayed CE (**17.0x** validation noise floor).
   - At 750M tokens: 6L leads by **-0.1371** decayed CE (**18.0x** validation noise floor; -0.0595 BPB).
   - **Trend**: The gap between 6L and 4L **widens monotonically** as token volume increases (-0.1191 -> -0.1290 -> -0.1371).

2. **Equal Wall-Clock Time (3.00 Hours)**:
   - In 3.00 hours, 4L processed 1,097.7M tokens (~1.10B), while 6L processed 800.1M tokens (37% fewer tokens due to lower training throughput: 74k vs 106k tok/s).
   - Despite seeing **297.6 million fewer tokens**, 6L achieves a **lower validation loss**:
     - 4L (3h decayed, 1.12B tokens): Stage A CE = **3.8475** (BPB = 1.6870)
     - 6L (3h decayed, 0.83B tokens): Stage A CE = **3.7470** (BPB = 1.6435)
     - **Delta (6L - 4L)**: **-0.1005 CE** (**13.2x** validation noise floor, **3.96x** training noise floor) and **-0.0435 BPB**.
   - On the neutral Wikipedia benchmark (al_wikipedia_neutral):
     - 4L (3h decayed): **4.4388**
     - 6L (3h decayed): **4.3707** (Delta = **-0.0681**).

3. **Inference Decode Speed vs Quality Trade-off**:
   - Single-stream decode throughput:
     - 4L_768H_1KV: **1,514.8 t/s** (0k depth) / **1,356.4 t/s** (32k depth)
     - 6L_768H_1KV: **1,326.5 t/s** (0k depth) / **1,180.7 t/s** (32k depth)
   - 4L is **+14.2% faster** at 0k and **+14.9% faster** at 32k.
   - However, 6L delivers **-0.1005 lower CE loss (-0.0435 BPB)** at equal training time and **-0.1371 lower CE loss (-0.0595 BPB)** at equal tokens.

4. **Hardware & Telemetry Verification**:
   - Continuous 60-second telemetry logging confirmed zero thermal throttling. GPU temperatures remained at 77-81°C (limit 83°C), SM clocks remained at 1,890 MHz, memory clocks remained at 9,251 MHz, and VRAM peaked at 6.5 GB (out of 8.0 GB).

---

## Complete Checkpoint Summary Table

| model | checkpoint | tokens | wall_clock_sec | is_decayed | stage_a_ce | stage_a_bpb | wiki_ce | wiki_bpb | cosmo_ce | fineweb_ce | tinystories_ce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4L_768H_1KV | 250M_undecayed | 250019840 | 2444.1 | False | 4.2141 | 1.8459 | 4.7442 | 2.4332 | 3.9076 | 4.7795 | 3.0487 |
| 4L_768H_1KV | 250M_decayed | 274989056 | 2444.1 | True | 4.0881 | 1.7912 | 4.6329 | 2.3761 | 3.7649 | 4.6723 | 2.9133 |
| 4L_768H_1KV | 500M_undecayed | 500039680 | 4955.1 | False | 4.0653 | 1.7814 | 4.622 | 2.3705 | 3.7354 | 4.6532 | 2.9047 |
| 4L_768H_1KV | 500M_decayed | 525008896 | 4955.1 | True | 3.9528 | 1.7327 | 4.5227 | 2.3195 | 3.6063 | 4.5589 | 2.7844 |
| 4L_768H_1KV | 750M_undecayed | 750059520 | 7377.4 | False | 4.0001 | 1.7532 | 4.556 | 2.3366 | 3.6588 | 4.6005 | 2.834 |
| 4L_768H_1KV | 750M_decayed | 775028736 | 7377.4 | True | 3.8937 | 1.7071 | 4.476 | 2.2956 | 3.5373 | 4.5092 | 2.7274 |
| 4L_768H_1KV | 3hr_undecayed | 1097662464 | 10800.1 | False | 3.948 | 1.7306 | 4.5413 | 2.3291 | 3.595 | 4.5577 | 2.7925 |
| 4L_768H_1KV | 3hr_decayed | 1122631680 | 10800.1 | True | 3.8475 | 1.687 | 4.4388 | 2.2765 | 3.4831 | 4.4697 | 2.6871 |
| 6L_768H_1KV | 250M_undecayed | 250019840 | 3223.7 | False | 4.1015 | 1.7971 | 4.6512 | 2.3855 | 3.7775 | 4.6839 | 2.939 |
| 6L_768H_1KV | 250M_decayed | 274989056 | 3223.7 | True | 3.969 | 1.7396 | 4.546 | 2.3315 | 3.6286 | 4.5687 | 2.8018 |
| 6L_768H_1KV | 500M_undecayed | 500039680 | 6664.7 | False | 3.9376 | 1.7261 | 4.5388 | 2.3278 | 3.5877 | 4.5432 | 2.7872 |
| 6L_768H_1KV | 500M_decayed | 525008896 | 6664.7 | True | 3.8238 | 1.6767 | 4.4302 | 2.2721 | 3.4592 | 4.4454 | 2.6668 |
| 6L_768H_1KV | 750M_undecayed | 750059520 | 10115.9 | False | 3.8666 | 1.6953 | 4.4741 | 2.2946 | 3.5059 | 4.4848 | 2.7082 |
| 6L_768H_1KV | 750M_decayed | 775028736 | 10115.9 | True | 3.7566 | 1.6476 | 4.3745 | 2.2435 | 3.3792 | 4.3899 | 2.6052 |
| 6L_768H_1KV | 3hr_undecayed | 800063488 | 10800.4 | False | 3.8571 | 1.6912 | 4.4734 | 2.2943 | 3.4914 | 4.478 | 2.7088 |
| 6L_768H_1KV | 3hr_decayed | 825032704 | 10800.4 | True | 3.747 | 1.6435 | 4.3707 | 2.2416 | 3.3682 | 4.381 | 2.5989 |

---

## Complete Gap Table (6L minus 4L)

| checkpoint | is_decayed | 4L_stage_a_ce | 6L_stage_a_ce | delta_ce (6L - 4L) | delta_bpb (6L - 4L) | delta_wiki_ce | delta_cosmo_ce | delta_fineweb_ce | delta_tinystories_ce | multiple_of_noise_val (0.0076) | multiple_of_noise_train (0.0254) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250M_undecayed | False | 4.2141 | 4.1015 | -0.1126 | -0.0488 | -0.093 | -0.1301 | -0.0956 | -0.1097 | 14.82 | 4.43 |
| 250M_decayed | True | 4.0881 | 3.969 | -0.1191 | -0.0516 | -0.0869 | -0.1363 | -0.1036 | -0.1115 | 15.67 | 4.69 |
| 500M_undecayed | False | 4.0653 | 3.9376 | -0.1277 | -0.0553 | -0.0832 | -0.1477 | -0.11 | -0.1175 | 16.8 | 5.03 |
| 500M_decayed | True | 3.9528 | 3.8238 | -0.129 | -0.056 | -0.0925 | -0.1471 | -0.1135 | -0.1176 | 16.97 | 5.08 |
| 750M_undecayed | False | 4.0001 | 3.8666 | -0.1335 | -0.0579 | -0.0819 | -0.1529 | -0.1157 | -0.1258 | 17.57 | 5.26 |
| 750M_decayed | True | 3.8937 | 3.7566 | -0.1371 | -0.0595 | -0.1015 | -0.1581 | -0.1193 | -0.1222 | 18.04 | 5.4 |
| 3hr_undecayed | False | 3.948 | 3.8571 | -0.0909 | -0.0394 | -0.0679 | -0.1036 | -0.0797 | -0.0837 | 11.96 | 3.58 |
| 3hr_decayed | True | 3.8475 | 3.747 | -0.1005 | -0.0435 | -0.0681 | -0.1149 | -0.0887 | -0.0882 | 13.22 | 3.96 |
