# Dual geometry on identical pairs (language basis, mid layer)

Same single-token pairs, same null pool, same null draws on both sides. gap > 0 = further toward the head than random word groups. Bound ±0.082.

| model | concepts | gap vocabulary | gap activation | d = vocab − act |
|---|---|---|---|---|
| Qwen2.5-0.5B | 40 | +0.071 | +0.004 | +0.066 |
| Qwen2.5-3B | 40 | +0.118 | -0.005 | +0.124 |
| Qwen2.5-7B | 40 | +0.081 | -0.001 | +0.082 |
| Qwen3-4B | 40 | +0.115 | +0.001 | +0.114 |
| Mistral-7B-v0.3 | 38 | +0.128 | -0.006 | +0.134 |
| Llama-3.2-3B | 40 | +0.112 | -0.001 | +0.113 |
| Llama-3.1-8B | 40 | +0.090 | 0.000 | +0.091 |
| gemma-2-2b | 42 | +0.111 | -0.010 | +0.121 |

| quantity | models | n | mean | 95% CI | p (t) | 90% CI | p TOST | equivalent |
|---|---|---|---|---|---|---|---|---|
| **A: d (primary)** | all | 8 | +0.106 | [+0.086, +0.125] | <0.001 | [+0.090, +0.121] | 0.987 | no |
| **A: d (primary)** | without gemma | 7 | +0.103 | [+0.081, +0.126] | <0.001 | [+0.085, +0.121] | 0.970 | no |
| **B: activation gap (primary)** | all | 8 | -0.002 | [-0.006, +0.001] | 0.182 | [-0.006, +0.001] | <0.001 | yes |
| **B: activation gap (primary)** | without gemma | 7 | -0.001 | [-0.005, +0.002] | 0.380 | [-0.004, +0.001] | <0.001 | yes |
| vocabulary gap | all | 8 | +0.103 | [+0.086, +0.120] | <0.001 | [+0.090, +0.117] | 0.990 | no |
| vocabulary gap | without gemma | 7 | +0.102 | [+0.082, +0.122] | <0.001 | [+0.086, +0.118] | 0.976 | no |
