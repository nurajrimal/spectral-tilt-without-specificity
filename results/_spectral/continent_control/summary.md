# Semantic (continent) control for the quiet tail

Equivalence bound ±0.1 retention units. Δ > 0 means truth prefers the quiet tail MORE than the control does.

Primary: truth_cities, mid layer, control named_continent.

## Per model

| model | dataset | layer | control | Δ | 95% CI | concept headroom | control headroom | leak | usable |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | truth_cities | mid | named_continent | +0.028 | [+0.008, +0.047] | +0.497 | +0.345 | 0.495 | yes |
| Llama-3.1-8B | truth_cities | mid | city_continent | +0.028 | [+0.008, +0.047] | +0.497 | +0.277 | 0.495 | yes |
| Llama-3.1-8B | truth_cities | mid | length_bucket | -0.111 | [-0.214, -0.005] | +0.497 | +0.347 | 0.495 | yes |
| Llama-3.1-8B | truth_cities | final | named_continent | -0.110 | [-0.161, -0.058] | +0.492 | +0.338 | 0.498 | yes |
| Llama-3.1-8B | truth_cities | final | city_continent | -0.107 | [-0.156, -0.059] | +0.492 | +0.331 | 0.498 | yes |
| Llama-3.1-8B | truth_cities | final | length_bucket | -0.110 | [-0.229, +0.008] | +0.492 | +0.310 | 0.498 | yes |
| Llama-3.1-8B | truth_cities_negated | mid | named_continent | -0.004 | [-0.016, +0.008] | +0.499 | +0.349 | 0.506 | yes |
| Llama-3.1-8B | truth_cities_negated | mid | city_continent | -0.004 | [-0.016, +0.008] | +0.499 | +0.315 | 0.506 | yes |
| Llama-3.1-8B | truth_cities_negated | mid | length_bucket | -0.060 | [-0.154, +0.035] | +0.499 | +0.350 | 0.506 | yes |
| Llama-3.1-8B | truth_cities_negated | final | named_continent | +0.003 | [-0.039, +0.045] | +0.499 | +0.331 | 0.480 | yes |
| Llama-3.1-8B | truth_cities_negated | final | city_continent | +0.003 | [-0.039, +0.045] | +0.499 | +0.313 | 0.480 | yes |
| Llama-3.1-8B | truth_cities_negated | final | length_bucket | +0.057 | [-0.049, +0.166] | +0.499 | +0.291 | 0.480 | yes |
| Llama-3.2-3B | truth_cities | mid | named_continent | +0.073 | [+0.047, +0.101] | +0.495 | +0.353 | 0.504 | yes |
| Llama-3.2-3B | truth_cities | mid | city_continent | +0.073 | [+0.047, +0.101] | +0.495 | +0.270 | 0.504 | yes |
| Llama-3.2-3B | truth_cities | mid | length_bucket | +0.098 | [+0.003, +0.190] | +0.495 | +0.351 | 0.504 | yes |
| Llama-3.2-3B | truth_cities | final | named_continent | +0.116 | [+0.059, +0.172] | +0.495 | +0.335 | 0.477 | yes |
| Llama-3.2-3B | truth_cities | final | city_continent | +0.116 | [+0.059, +0.172] | +0.495 | +0.325 | 0.477 | yes |
| Llama-3.2-3B | truth_cities | final | length_bucket | +0.072 | [-0.058, +0.201] | +0.495 | +0.306 | 0.477 | yes |
| Llama-3.2-3B | truth_cities_negated | mid | named_continent | -0.006 | [-0.022, +0.010] | +0.498 | +0.337 | 0.494 | yes |
| Llama-3.2-3B | truth_cities_negated | mid | city_continent | -0.006 | [-0.022, +0.010] | +0.498 | +0.298 | 0.494 | yes |
| Llama-3.2-3B | truth_cities_negated | mid | length_bucket | -0.261 | [-0.361, -0.162] | +0.498 | +0.329 | 0.494 | yes |
| Llama-3.2-3B | truth_cities_negated | final | named_continent | -0.004 | [-0.047, +0.039] | +0.499 | +0.328 | 0.494 | yes |
| Llama-3.2-3B | truth_cities_negated | final | city_continent | -0.026 | [-0.074, +0.021] | +0.499 | +0.312 | 0.494 | yes |
| Llama-3.2-3B | truth_cities_negated | final | length_bucket | -0.080 | [-0.226, +0.073] | +0.499 | +0.282 | 0.494 | yes |
| Mistral-7B-v0.3 | truth_cities | mid | named_continent | +0.047 | [+0.027, +0.069] | +0.498 | +0.359 | 0.489 | yes |
| Mistral-7B-v0.3 | truth_cities | mid | city_continent | +0.047 | [+0.027, +0.069] | +0.498 | +0.313 | 0.489 | yes |
| Mistral-7B-v0.3 | truth_cities | mid | length_bucket | +0.023 | [-0.070, +0.115] | +0.498 | +0.359 | 0.489 | yes |
| Mistral-7B-v0.3 | truth_cities | final | named_continent | -0.006 | [-0.039, +0.028] | +0.495 | +0.347 | 0.502 | yes |
| Mistral-7B-v0.3 | truth_cities | final | city_continent | -0.006 | [-0.039, +0.028] | +0.495 | +0.296 | 0.502 | yes |
| Mistral-7B-v0.3 | truth_cities | final | length_bucket | -0.044 | [-0.156, +0.069] | +0.495 | +0.310 | 0.502 | yes |
| Mistral-7B-v0.3 | truth_cities_negated | mid | named_continent | -0.002 | [-0.020, +0.014] | +0.499 | +0.348 | 0.496 | yes |
| Mistral-7B-v0.3 | truth_cities_negated | mid | city_continent | -0.017 | [-0.039, +0.004] | +0.499 | +0.322 | 0.496 | yes |
| Mistral-7B-v0.3 | truth_cities_negated | mid | length_bucket | -0.369 | [-0.473, -0.267] | +0.499 | +0.349 | 0.496 | yes |
| Mistral-7B-v0.3 | truth_cities_negated | final | named_continent | +0.042 | [+0.009, +0.075] | +0.499 | +0.321 | 0.513 | yes |
| Mistral-7B-v0.3 | truth_cities_negated | final | city_continent | +0.030 | [-0.004, +0.064] | +0.499 | +0.328 | 0.513 | yes |
| Mistral-7B-v0.3 | truth_cities_negated | final | length_bucket | -0.042 | [-0.156, +0.072] | +0.499 | +0.313 | 0.513 | yes |
| Qwen2.5-0.5B | truth_cities | mid | named_continent | +0.132 | [+0.072, +0.189] | +0.460 | +0.310 | 0.543 | yes |
| Qwen2.5-0.5B | truth_cities | mid | city_continent | +0.132 | [+0.072, +0.189] | +0.460 | +0.176 | 0.543 | yes |
| Qwen2.5-0.5B | truth_cities | mid | length_bucket | +0.170 | [+0.056, +0.284] | +0.460 | +0.327 | 0.543 | yes |
| Qwen2.5-0.5B | truth_cities | final | named_continent | +0.174 | [+0.084, +0.264] | +0.443 | +0.321 | 0.475 | yes |
| Qwen2.5-0.5B | truth_cities | final | city_continent | +0.174 | [+0.084, +0.264] | +0.443 | +0.231 | 0.475 | yes |
| Qwen2.5-0.5B | truth_cities | final | length_bucket | +0.084 | [-0.092, +0.254] | +0.443 | +0.283 | 0.475 | yes |
| Qwen2.5-0.5B | truth_cities_negated | mid | named_continent | +0.387 | [+0.325, +0.451] | +0.457 | +0.326 | 0.498 | yes |
| Qwen2.5-0.5B | truth_cities_negated | mid | city_continent | +0.387 | [+0.325, +0.451] | +0.457 | +0.184 | 0.498 | yes |
| Qwen2.5-0.5B | truth_cities_negated | mid | length_bucket | +0.453 | [+0.305, +0.600] | +0.457 | +0.312 | 0.498 | yes |
| Qwen2.5-0.5B | truth_cities_negated | final | named_continent | +0.104 | [+0.013, +0.195] | +0.441 | +0.295 | 0.463 | yes |
| Qwen2.5-0.5B | truth_cities_negated | final | city_continent | +0.104 | [+0.013, +0.195] | +0.441 | +0.237 | 0.463 | yes |
| Qwen2.5-0.5B | truth_cities_negated | final | length_bucket | +0.049 | [-0.119, +0.216] | +0.441 | +0.269 | 0.463 | yes |
| Qwen2.5-3B | truth_cities | mid | named_continent | -0.139 | [-0.183, -0.097] | +0.497 | +0.327 | 0.528 | yes |
| Qwen2.5-3B | truth_cities | mid | city_continent | -0.145 | [-0.189, -0.104] | +0.497 | +0.285 | 0.528 | yes |
| Qwen2.5-3B | truth_cities | mid | length_bucket | -0.107 | [-0.203, -0.014] | +0.497 | +0.360 | 0.528 | yes |
| Qwen2.5-3B | truth_cities | final | named_continent | -0.124 | [-0.172, -0.078] | +0.488 | +0.341 | 0.520 | yes |
| Qwen2.5-3B | truth_cities | final | city_continent | -0.124 | [-0.172, -0.078] | +0.488 | +0.281 | 0.520 | yes |
| Qwen2.5-3B | truth_cities | final | length_bucket | -0.090 | [-0.200, +0.018] | +0.488 | +0.291 | 0.520 | yes |
| Qwen2.5-3B | truth_cities_negated | mid | named_continent | -0.169 | [-0.217, -0.121] | +0.498 | +0.333 | 0.519 | yes |
| Qwen2.5-3B | truth_cities_negated | mid | city_continent | -0.186 | [-0.233, -0.139] | +0.498 | +0.269 | 0.519 | yes |
| Qwen2.5-3B | truth_cities_negated | mid | length_bucket | -0.186 | [-0.303, -0.066] | +0.498 | +0.350 | 0.519 | yes |
| Qwen2.5-3B | truth_cities_negated | final | named_continent | -0.105 | [-0.160, -0.051] | +0.493 | +0.357 | 0.528 | yes |
| Qwen2.5-3B | truth_cities_negated | final | city_continent | -0.105 | [-0.160, -0.051] | +0.493 | +0.237 | 0.528 | yes |
| Qwen2.5-3B | truth_cities_negated | final | length_bucket | -0.190 | [-0.309, -0.069] | +0.493 | +0.300 | 0.528 | yes |
| Qwen2.5-7B | truth_cities | mid | named_continent | -0.025 | [-0.056, +0.006] | +0.496 | +0.352 | 0.478 | yes |
| Qwen2.5-7B | truth_cities | mid | city_continent | -0.014 | [-0.043, +0.016] | +0.496 | +0.298 | 0.478 | yes |
| Qwen2.5-7B | truth_cities | mid | length_bucket | -0.051 | [-0.135, +0.036] | +0.496 | +0.370 | 0.478 | yes |
| Qwen2.5-7B | truth_cities | final | named_continent | -0.068 | [-0.099, -0.036] | +0.491 | +0.328 | 0.503 | yes |
| Qwen2.5-7B | truth_cities | final | city_continent | -0.068 | [-0.099, -0.036] | +0.491 | +0.272 | 0.503 | yes |
| Qwen2.5-7B | truth_cities | final | length_bucket | -0.051 | [-0.192, +0.090] | +0.491 | +0.297 | 0.503 | yes |
| Qwen2.5-7B | truth_cities_negated | mid | named_continent | +0.051 | [+0.014, +0.090] | +0.499 | +0.362 | 0.483 | yes |
| Qwen2.5-7B | truth_cities_negated | mid | city_continent | +0.057 | [+0.020, +0.094] | +0.499 | +0.294 | 0.483 | yes |
| Qwen2.5-7B | truth_cities_negated | mid | length_bucket | +0.173 | [+0.065, +0.280] | +0.499 | +0.352 | 0.483 | yes |
| Qwen2.5-7B | truth_cities_negated | final | named_continent | +0.088 | [+0.020, +0.155] | +0.498 | +0.356 | 0.536 | yes |
| Qwen2.5-7B | truth_cities_negated | final | city_continent | +0.088 | [+0.020, +0.155] | +0.498 | +0.278 | 0.536 | yes |
| Qwen2.5-7B | truth_cities_negated | final | length_bucket | +0.098 | [-0.016, +0.209] | +0.498 | +0.303 | 0.536 | yes |
| Qwen3-4B | truth_cities | mid | named_continent | -0.026 | [-0.052, +0.000] | +0.492 | +0.348 | 0.483 | yes |
| Qwen3-4B | truth_cities | mid | city_continent | -0.026 | [-0.052, +0.000] | +0.492 | +0.261 | 0.483 | yes |
| Qwen3-4B | truth_cities | mid | length_bucket | -0.050 | [-0.156, +0.055] | +0.492 | +0.362 | 0.483 | yes |
| Qwen3-4B | truth_cities | final | named_continent | -0.020 | [-0.050, +0.010] | +0.484 | +0.284 | 0.494 | yes |
| Qwen3-4B | truth_cities | final | city_continent | -0.020 | [-0.050, +0.010] | +0.484 | +0.334 | 0.494 | yes |
| Qwen3-4B | truth_cities | final | length_bucket | -0.218 | [-0.332, -0.104] | +0.484 | +0.306 | 0.494 | yes |
| Qwen3-4B | truth_cities_negated | mid | named_continent | -0.123 | [-0.159, -0.089] | +0.492 | +0.345 | 0.502 | yes |
| Qwen3-4B | truth_cities_negated | mid | city_continent | -0.123 | [-0.159, -0.089] | +0.492 | +0.258 | 0.502 | yes |
| Qwen3-4B | truth_cities_negated | mid | length_bucket | -0.099 | [-0.205, +0.007] | +0.492 | +0.362 | 0.502 | yes |
| Qwen3-4B | truth_cities_negated | final | named_continent | -0.122 | [-0.170, -0.074] | +0.489 | +0.299 | 0.501 | yes |
| Qwen3-4B | truth_cities_negated | final | city_continent | -0.122 | [-0.170, -0.074] | +0.489 | +0.314 | 0.501 | yes |
| Qwen3-4B | truth_cities_negated | final | length_bucket | -0.220 | [-0.337, -0.104] | +0.489 | +0.278 | 0.501 | yes |
| gemma-2-2b | truth_cities | mid | named_continent | +0.339 | [+0.199, +0.475] | +0.230 | +0.093 | 0.509 | yes |
| gemma-2-2b | truth_cities | mid | city_continent | +0.339 | [+0.199, +0.475] | +0.230 | +0.183 | 0.509 | yes |
| gemma-2-2b | truth_cities | mid | length_bucket | +0.168 | [-0.012, +0.350] | +0.230 | +0.234 | 0.509 | yes |
| gemma-2-2b | truth_cities | final | named_continent | -0.058 | [-0.266, +0.145] | +0.202 | +0.135 | 0.488 | yes |
| gemma-2-2b | truth_cities | final | city_continent | -0.058 | [-0.266, +0.145] | +0.202 | +0.291 | 0.488 | yes |
| gemma-2-2b | truth_cities | final | length_bucket | -0.091 | [-0.320, +0.133] | +0.202 | +0.236 | 0.488 | yes |
| gemma-2-2b | truth_cities_negated | mid | named_continent | +0.167 | [-0.004, +0.339] | +0.228 | +0.109 | 0.485 | yes |
| gemma-2-2b | truth_cities_negated | mid | city_continent | +0.167 | [-0.004, +0.339] | +0.228 | +0.224 | 0.485 | yes |
| gemma-2-2b | truth_cities_negated | mid | length_bucket | +0.146 | [-0.056, +0.348] | +0.228 | +0.225 | 0.485 | yes |
| gemma-2-2b | truth_cities_negated | final | named_continent | +0.251 | [+0.093, +0.419] | +0.210 | +0.121 | 0.508 | yes |
| gemma-2-2b | truth_cities_negated | final | city_continent | +0.251 | [+0.093, +0.419] | +0.210 | +0.297 | 0.508 | yes |
| gemma-2-2b | truth_cities_negated | final | length_bucket | +0.241 | [+0.046, +0.442] | +0.210 | +0.198 | 0.508 | yes |

## Across models (usable cells only)

| dataset | layer | control | models | n | mean Δ | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| truth_cities | mid | **named_continent (primary)** | all | 8 | +0.054 | [-0.064, +0.171] | 0.317 | 0.312 | [-0.041, +0.148] | 0.190 | no |
| truth_cities | mid | **named_continent (primary)** | without gemma | 7 | +0.013 | [-0.068, +0.093] | 0.712 | 0.578 | [-0.051, +0.076] | 0.019 | yes |
| truth_cities | mid | city_continent | all | 8 | +0.054 | [-0.064, +0.172] | 0.313 | 0.312 | [-0.040, +0.149] | 0.194 | no |
| truth_cities | mid | city_continent | without gemma | 7 | +0.013 | [-0.068, +0.095] | 0.699 | 0.578 | [-0.051, +0.078] | 0.020 | yes |
| truth_cities | mid | length_bucket | all | 8 | +0.018 | [-0.079, +0.114] | 0.681 | 0.844 | [-0.060, +0.095] | 0.042 | yes |
| truth_cities | mid | length_bucket | without gemma | 7 | -0.004 | [-0.102, +0.094] | 0.925 | 0.812 | [-0.082, +0.074] | 0.027 | yes |
| truth_cities | final | named_continent | all | 8 | -0.012 | [-0.100, +0.077] | 0.760 | 0.641 | [-0.083, +0.059] | 0.025 | yes |
| truth_cities | final | named_continent | without gemma | 7 | -0.005 | [-0.110, +0.099] | 0.904 | 0.812 | [-0.088, +0.077] | 0.034 | yes |
| truth_cities | final | city_continent | all | 8 | -0.012 | [-0.100, +0.077] | 0.766 | 0.641 | [-0.082, +0.059] | 0.025 | yes |
| truth_cities | final | city_continent | without gemma | 7 | -0.005 | [-0.109, +0.099] | 0.911 | 0.812 | [-0.087, +0.077] | 0.033 | yes |
| truth_cities | final | length_bucket | all | 8 | -0.056 | [-0.138, +0.026] | 0.151 | 0.148 | [-0.122, +0.010] | 0.124 | no |
| truth_cities | final | length_bucket | without gemma | 7 | -0.051 | [-0.148, +0.046] | 0.246 | 0.297 | [-0.128, +0.026] | 0.133 | no |
| truth_cities_negated | mid | named_continent | all | 8 | +0.038 | [-0.108, +0.183] | 0.559 | 1.000 | [-0.079, +0.154] | 0.173 | no |
| truth_cities_negated | mid | named_continent | without gemma | 7 | +0.019 | [-0.147, +0.185] | 0.786 | 0.688 | [-0.113, +0.151] | 0.139 | no |
| truth_cities_negated | mid | city_continent | all | 8 | +0.034 | [-0.114, +0.183] | 0.602 | 1.000 | [-0.085, +0.154] | 0.166 | no |
| truth_cities_negated | mid | city_continent | without gemma | 7 | +0.015 | [-0.154, +0.185] | 0.832 | 0.688 | [-0.119, +0.150] | 0.134 | no |
| truth_cities_negated | mid | length_bucket | all | 8 | -0.025 | [-0.250, +0.199] | 0.797 | 0.742 | [-0.205, +0.154] | 0.229 | no |
| truth_cities_negated | mid | length_bucket | without gemma | 7 | -0.050 | [-0.309, +0.209] | 0.654 | 0.578 | [-0.255, +0.156] | 0.326 | no |
| truth_cities_negated | final | named_continent | all | 8 | +0.032 | [-0.068, +0.133] | 0.473 | 0.742 | [-0.048, +0.113] | 0.077 | no |
| truth_cities_negated | final | named_continent | without gemma | 7 | +0.001 | [-0.080, +0.082] | 0.979 | 0.938 | [-0.064, +0.065] | 0.012 | yes |
| truth_cities_negated | final | city_continent | all | 8 | +0.028 | [-0.073, +0.129] | 0.534 | 0.742 | [-0.053, +0.109] | 0.068 | no |
| truth_cities_negated | final | city_continent | without gemma | 7 | -0.004 | [-0.085, +0.077] | 0.911 | 0.938 | [-0.068, +0.060] | 0.014 | yes |
| truth_cities_negated | final | length_bucket | all | 8 | -0.011 | [-0.139, +0.118] | 0.848 | 1.000 | [-0.114, +0.092] | 0.072 | no |
| truth_cities_negated | final | length_bucket | without gemma | 7 | -0.047 | [-0.162, +0.068] | 0.357 | 0.578 | [-0.138, +0.044] | 0.150 | no |

Reading the primary row (the pre-registered decision rule):

- 95% CI above 0 in both versions: truth is quiet-tail-specific against a semantic control; the length result does not generalise.
- 90% CI inside ±0.10 in both versions: not specific even against a semantic control.
- otherwise: inconclusive at this n.

`leak` is the full-dimensional accuracy of pair_id_random and must sit near 0.5; a cell where it does not is not trustworthy.
