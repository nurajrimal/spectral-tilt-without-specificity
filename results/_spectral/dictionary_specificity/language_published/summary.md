# Dictionary-side specificity (language basis, token rule published)

gap > 0: the concept direction built from unembedding rows sits further toward the HEAD than the null. Equivalence bound ±0.082 Gini units.

## Per model (mean over concepts)

| model | concepts | Gini | gap lexical | sig (head) | gap vocab_n | gap token_pair | gap repartition | gap isotropic |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 40/42 | +0.014 | +0.070 | 24 (21) | +0.111 | +0.109 | +0.080 | +0.175 |
| Qwen2.5-3B | 40/42 | +0.066 | +0.118 | 29 (24) | +0.175 | +0.177 | +0.134 | +0.240 |
| Qwen2.5-7B | 40/42 | -0.007 | +0.081 | 25 (22) | +0.121 | +0.118 | +0.085 | +0.179 |
| Qwen3-4B | 40/42 | +0.052 | +0.115 | 31 (24) | +0.182 | +0.185 | +0.129 | +0.238 |
| Mistral-7B-v0.3 | 38/42 | +0.054 | +0.128 | 29 (23) | +0.156 | +0.161 | +0.150 | +0.270 |
| Llama-3.2-3B | 40/42 | +0.042 | +0.112 | 28 (24) | +0.177 | +0.179 | +0.129 | +0.247 |
| Llama-3.1-8B | 40/42 | +0.005 | +0.091 | 26 (22) | +0.139 | +0.139 | +0.098 | +0.200 |
| gemma-2-2b | 42/42 | +0.028 | +0.111 | 25 (24) | +0.146 | +0.145 | +0.125 | +0.261 |

## Across models

| null | models | n | mean gap | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |
|---|---|---|---|---|---|---|---|---|---|
| **lexical (primary)** | all | 8 | +0.103 | [+0.086, +0.120] | <0.001 | 0.008 | [+0.090, +0.117] | 0.990 | no |
| **lexical (primary)** | without gemma | 7 | +0.102 | [+0.082, +0.122] | <0.001 | 0.016 | [+0.086, +0.118] | 0.976 | no |
| repartition | all | 8 | +0.116 | [+0.095, +0.138] | <0.001 | 0.008 | [+0.100, +0.133] | 0.997 | no |
| repartition | without gemma | 7 | +0.115 | [+0.090, +0.140] | <0.001 | 0.016 | [+0.095, +0.135] | 0.991 | no |
| vocab_n | all | 8 | +0.151 | [+0.129, +0.173] | <0.001 | 0.008 | [+0.133, +0.169] | 1.000 | no |
| vocab_n | without gemma | 7 | +0.152 | [+0.125, +0.178] | <0.001 | 0.016 | [+0.131, +0.173] | 1.000 | no |
| token_pair | all | 8 | +0.152 | [+0.128, +0.176] | <0.001 | 0.008 | [+0.133, +0.171] | 1.000 | no |
| token_pair | without gemma | 7 | +0.153 | [+0.124, +0.181] | <0.001 | 0.016 | [+0.130, +0.175] | 1.000 | no |
| isotropic | all | 8 | +0.226 | [+0.195, +0.257] | <0.001 | 0.008 | [+0.202, +0.251] | 1.000 | no |
| isotropic | without gemma | 7 | +0.221 | [+0.187, +0.255] | <0.001 | 0.016 | [+0.194, +0.248] | 1.000 | no |

**Primary verdict (lexical null, frozen rule): SPECIFIC (HEAD)**

Published comparison (concept vs one random token pair): concept further toward the head in 8 of 8 models (reproduces; gate: at least 6).

## Dual contrast (descriptive; activation half from word_specificity, final layer, all concepts)

| model | dictionary gap (lexical) | activation gap (lexical) |
|---|---|---|
| Qwen2.5-0.5B | +0.070 | +0.008 |
| Qwen2.5-3B | +0.118 | +0.004 |
| Qwen2.5-7B | +0.081 | +0.013 |
| Qwen3-4B | +0.115 | -0.004 |
| Mistral-7B-v0.3 | +0.128 | +0.015 |
| Llama-3.2-3B | +0.112 | +0.021 |
| Llama-3.1-8B | +0.091 | -0.018 |
| gemma-2-2b | +0.111 | +0.007 |
