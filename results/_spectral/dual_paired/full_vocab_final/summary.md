# Dual geometry on identical pairs (full_vocab basis, final layer)

Same single-token pairs, same null pool, same null draws on both sides. gap > 0 = further toward the head than random word groups. Bound ±0.082.

| model | concepts | gap vocabulary | gap activation | d = vocab − act |
|---|---|---|---|---|
| Qwen2.5-0.5B | 40 | -0.002 | -0.002 | 0.000 |
| Qwen2.5-3B | 40 | +0.076 | -0.004 | +0.081 |
| Qwen2.5-7B | 40 | +0.020 | +0.004 | +0.016 |
| Qwen3-4B | 40 | +0.075 | -0.007 | +0.082 |
| Mistral-7B-v0.3 | 38 | +0.107 | +0.005 | +0.102 |
| Llama-3.2-3B | 40 | +0.083 | +0.014 | +0.069 |
| Llama-3.1-8B | 40 | +0.052 | -0.022 | +0.074 |
| gemma-2-2b | 42 | +0.064 | +0.008 | +0.056 |

| quantity | models | n | mean | 95% CI | p (t) | 90% CI | p TOST | equivalent |
|---|---|---|---|---|---|---|---|---|
| **A: d (primary)** | all | 8 | +0.060 | [+0.031, +0.089] | 0.002 | [+0.037, +0.083] | 0.059 | no |
| **A: d (primary)** | without gemma | 7 | +0.061 | [+0.026, +0.095] | 0.005 | [+0.033, +0.088] | 0.092 | no |
| **B: activation gap (primary)** | all | 8 | -0.001 | [-0.010, +0.009] | 0.883 | [-0.008, +0.007] | <0.001 | yes |
| **B: activation gap (primary)** | without gemma | 7 | -0.002 | [-0.012, +0.009] | 0.693 | [-0.010, +0.007] | <0.001 | yes |
| vocabulary gap | all | 8 | +0.059 | [+0.030, +0.089] | 0.002 | [+0.036, +0.083] | 0.058 | no |
| vocabulary gap | without gemma | 7 | +0.059 | [+0.023, +0.094] | 0.007 | [+0.031, +0.087] | 0.080 | no |
