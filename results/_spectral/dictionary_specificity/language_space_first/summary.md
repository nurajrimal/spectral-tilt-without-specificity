# Dictionary-side specificity (language basis, token rule space_first)

gap > 0: the concept direction built from unembedding rows sits further toward the HEAD than the null. Equivalence bound ±0.082 Gini units.

## Per model (mean over concepts)

| model | concepts | Gini | gap lexical | sig (head) | gap vocab_n | gap token_pair | gap repartition | gap isotropic |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 40/42 | +0.038 | +0.090 | 25 (22) | +0.135 | +0.133 | +0.106 | +0.199 |
| Qwen2.5-3B | 40/42 | +0.088 | +0.133 | 28 (24) | +0.198 | +0.200 | +0.154 | +0.262 |
| Qwen2.5-7B | 40/42 | +0.042 | +0.120 | 31 (27) | +0.170 | +0.167 | +0.131 | +0.228 |
| Qwen3-4B | 40/42 | +0.082 | +0.133 | 31 (24) | +0.212 | +0.215 | +0.152 | +0.268 |
| Mistral-7B-v0.3 | 38/42 | +0.054 | +0.128 | 29 (23) | +0.156 | +0.161 | +0.150 | +0.270 |
| Llama-3.2-3B | 40/42 | +0.086 | +0.137 | 30 (25) | +0.221 | +0.223 | +0.161 | +0.291 |
| Llama-3.1-8B | 40/42 | +0.054 | +0.132 | 29 (25) | +0.189 | +0.189 | +0.148 | +0.249 |
| gemma-2-2b | 42/42 | +0.090 | +0.147 | 28 (27) | +0.208 | +0.207 | +0.171 | +0.323 |

## Across models

| null | models | n | mean gap | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |
|---|---|---|---|---|---|---|---|---|---|
| **lexical (primary)** | all | 8 | +0.127 | [+0.113, +0.142] | <0.001 | 0.008 | [+0.116, +0.139] | 1.000 | no |
| **lexical (primary)** | without gemma | 7 | +0.125 | [+0.110, +0.140] | <0.001 | 0.016 | [+0.113, +0.137] | 1.000 | no |
| repartition | all | 8 | +0.147 | [+0.130, +0.163] | <0.001 | 0.008 | [+0.133, +0.160] | 1.000 | no |
| repartition | without gemma | 7 | +0.143 | [+0.126, +0.161] | <0.001 | 0.016 | [+0.129, +0.157] | 1.000 | no |
| vocab_n | all | 8 | +0.186 | [+0.161, +0.211] | <0.001 | 0.008 | [+0.166, +0.206] | 1.000 | no |
| vocab_n | without gemma | 7 | +0.183 | [+0.154, +0.212] | <0.001 | 0.016 | [+0.160, +0.206] | 1.000 | no |
| token_pair | all | 8 | +0.187 | [+0.161, +0.213] | <0.001 | 0.008 | [+0.166, +0.207] | 1.000 | no |
| token_pair | without gemma | 7 | +0.184 | [+0.154, +0.213] | <0.001 | 0.016 | [+0.160, +0.207] | 1.000 | no |
| isotropic | all | 8 | +0.261 | [+0.230, +0.293] | <0.001 | 0.008 | [+0.236, +0.287] | 1.000 | no |
| isotropic | without gemma | 7 | +0.252 | [+0.224, +0.281] | <0.001 | 0.016 | [+0.230, +0.275] | 1.000 | no |

Published comparison (concept vs one random token pair): concept further toward the head in 8 of 8 models (reproduces; gate: at least 6).

## Dual contrast (descriptive; activation half from word_specificity, final layer, all concepts)

| model | dictionary gap (lexical) | activation gap (lexical) |
|---|---|---|
| Qwen2.5-0.5B | +0.090 | +0.008 |
| Qwen2.5-3B | +0.133 | +0.004 |
| Qwen2.5-7B | +0.120 | +0.013 |
| Qwen3-4B | +0.133 | -0.004 |
| Mistral-7B-v0.3 | +0.128 | +0.015 |
| Llama-3.2-3B | +0.137 | +0.021 |
| Llama-3.1-8B | +0.132 | -0.018 |
| gemma-2-2b | +0.147 | +0.007 |
