# Dual geometry on identical pairs (language basis, final layer)

Same single-token pairs, same null pool, same null draws on both sides. gap > 0 = further toward the head than random word groups. Bound ±0.082.

| model | concepts | gap vocabulary | gap activation | d = vocab − act |
|---|---|---|---|---|
| Qwen2.5-0.5B | 40 | +0.071 | +0.004 | +0.066 |
| Qwen2.5-3B | 40 | +0.118 | -0.001 | +0.119 |
| Qwen2.5-7B | 40 | +0.081 | +0.006 | +0.075 |
| Qwen3-4B | 40 | +0.115 | -0.005 | +0.120 |
| Mistral-7B-v0.3 | 38 | +0.128 | +0.007 | +0.121 |
| Llama-3.2-3B | 40 | +0.112 | +0.019 | +0.093 |
| Llama-3.1-8B | 40 | +0.090 | -0.011 | +0.101 |
| gemma-2-2b | 42 | +0.111 | +0.009 | +0.102 |

| quantity | models | n | mean | 95% CI | p (t) | 90% CI | p TOST | equivalent |
|---|---|---|---|---|---|---|---|---|
| **A: d (primary)** | all | 8 | +0.100 | [+0.082, +0.117] | <0.001 | [+0.086, +0.114] | 0.976 | no |
| **A: d (primary)** | without gemma | 7 | +0.099 | [+0.078, +0.120] | <0.001 | [+0.083, +0.116] | 0.956 | no |
| **B: activation gap (primary)** | all | 8 | +0.004 | [-0.004, +0.011] | 0.312 | [-0.003, +0.010] | <0.001 | yes |
| **B: activation gap (primary)** | without gemma | 7 | +0.003 | [-0.006, +0.012] | 0.469 | [-0.004, +0.010] | <0.001 | yes |
| vocabulary gap | all | 8 | +0.103 | [+0.086, +0.120] | <0.001 | [+0.090, +0.117] | 0.990 | no |
| vocabulary gap | without gemma | 7 | +0.102 | [+0.082, +0.122] | <0.001 | [+0.086, +0.118] | 0.976 | no |

**Primary verdict (frozen rule): ASYMMETRY CONFIRMED**
