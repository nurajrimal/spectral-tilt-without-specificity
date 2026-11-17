# Dictionary-side specificity (full_vocab basis, token rule published)

gap > 0: the concept direction built from unembedding rows sits further toward the HEAD than the null. Equivalence bound ±0.082 Gini units.

## Per model (mean over concepts)

| model | concepts | Gini | gap lexical | sig (head) | gap vocab_n | gap token_pair | gap repartition | gap isotropic |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 40/42 | -0.067 | -0.002 | 20 (11) | -0.006 | -0.009 | +0.003 | +0.078 |
| Qwen2.5-3B | 40/42 | +0.045 | +0.076 | 31 (28) | +0.078 | +0.077 | +0.092 | +0.178 |
| Qwen2.5-7B | 40/42 | -0.026 | +0.020 | 18 (15) | +0.006 | +0.006 | +0.025 | +0.104 |
| Qwen3-4B | 40/42 | +0.050 | +0.075 | 29 (27) | +0.086 | +0.085 | +0.090 | +0.180 |
| Mistral-7B-v0.3 | 38/42 | +0.034 | +0.107 | 26 (23) | +0.077 | +0.076 | +0.125 | +0.223 |
| Llama-3.2-3B | 40/42 | +0.042 | +0.083 | 31 (28) | +0.100 | +0.100 | +0.100 | +0.202 |
| Llama-3.1-8B | 40/42 | +0.018 | +0.052 | 31 (28) | +0.045 | +0.047 | +0.059 | +0.146 |
| gemma-2-2b | 42/42 | -0.001 | +0.064 | 21 (20) | +0.065 | +0.065 | +0.079 | +0.200 |

## Across models

| null | models | n | mean gap | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |
|---|---|---|---|---|---|---|---|---|---|
| **lexical (primary)** | all | 8 | +0.059 | [+0.030, +0.089] | 0.002 | 0.016 | [+0.036, +0.083] | 0.058 | no |
| **lexical (primary)** | without gemma | 7 | +0.059 | [+0.023, +0.094] | 0.007 | 0.031 | [+0.031, +0.087] | 0.081 | no |
| repartition | all | 8 | +0.072 | [+0.038, +0.106] | 0.002 | 0.008 | [+0.045, +0.099] | 0.248 | no |
| repartition | without gemma | 7 | +0.071 | [+0.030, +0.111] | 0.005 | 0.016 | [+0.039, +0.103] | 0.259 | no |
| vocab_n | all | 8 | +0.056 | [+0.024, +0.089] | 0.004 | 0.023 | [+0.031, +0.082] | 0.051 | no |
| vocab_n | without gemma | 7 | +0.055 | [+0.017, +0.093] | 0.012 | 0.047 | [+0.025, +0.086] | 0.068 | no |
| token_pair | all | 8 | +0.056 | [+0.023, +0.088] | 0.005 | 0.023 | [+0.030, +0.082] | 0.050 | yes |
| token_pair | without gemma | 7 | +0.054 | [+0.016, +0.093] | 0.014 | 0.047 | [+0.024, +0.085] | 0.066 | no |
| isotropic | all | 8 | +0.164 | [+0.121, +0.206] | <0.001 | 0.008 | [+0.130, +0.198] | 0.999 | no |
| isotropic | without gemma | 7 | +0.159 | [+0.110, +0.207] | <0.001 | 0.016 | [+0.120, +0.197] | 0.996 | no |

Published comparison (concept vs one random token pair): concept further toward the head in 7 of 8 models (reproduces; gate: at least 6).

## Dual contrast (descriptive; activation half from word_specificity, final layer, all concepts)

| model | dictionary gap (lexical) | activation gap (lexical) |
|---|---|---|
| Qwen2.5-0.5B | -0.002 | +0.002 |
| Qwen2.5-3B | +0.076 | +0.001 |
| Qwen2.5-7B | +0.020 | +0.008 |
| Qwen3-4B | +0.075 | -0.004 |
| Mistral-7B-v0.3 | +0.107 | +0.006 |
| Llama-3.2-3B | +0.083 | +0.016 |
| Llama-3.1-8B | +0.052 | -0.035 |
| gemma-2-2b | +0.064 | +0.007 |
