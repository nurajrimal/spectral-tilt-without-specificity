# EXPLORATORY: continent control, window-size sweep

Not pre-registered. At the pre-registered k = 0.5% the continent labels were unreadable at both ends, so Delta there is not a specificity statistic. A cell counts as READABLE when the control's retention exceeds 0.1 in the top or the bottom window; only readable, usable cells enter the across-model test.

| dataset | layer | control | k | readable models | n (all / w/o gemma) | mean Δ all | 95% CI | 90% CI | equiv. | mean Δ w/o gemma | 90% CI | equiv. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth_cities | final | city_continent | 0.25% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | final | city_continent | 0.5% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | final | city_continent | 1% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | final | city_continent | 2% | 5/8 | 5 / 4 | +0.038 | [-0.202, +0.279] | [-0.146, +0.223] | no | +0.083 | [-0.143, +0.309] | no |
| truth_cities | final | city_continent | 5% | 8/8 | 8 / 7 | +0.030 | [-0.147, +0.208] | [-0.112, +0.173] | no | +0.073 | [-0.067, +0.212] | no |
| truth_cities | final | city_continent | 10% | 8/8 | 8 / 7 | +0.043 | [-0.123, +0.209] | [-0.090, +0.176] | no | +0.085 | [-0.043, +0.212] | no |
| truth_cities | final | length_bucket | 0.25% | 8/8 | 8 / 7 | -0.033 | [-0.218, +0.152] | [-0.181, +0.115] | no | -0.073 | [-0.223, +0.077] | no |
| truth_cities | final | length_bucket | 0.5% | 8/8 | 8 / 7 | -0.056 | [-0.138, +0.026] | [-0.122, +0.010] | no | -0.051 | [-0.128, +0.026] | no |
| truth_cities | final | length_bucket | 1% | 8/8 | 8 / 7 | -0.017 | [-0.170, +0.137] | [-0.140, +0.106] | no | +0.048 | [+0.024, +0.071] | yes |
| truth_cities | final | length_bucket | 2% | 8/8 | 8 / 7 | +0.032 | [-0.058, +0.122] | [-0.040, +0.104] | no | +0.052 | [-0.021, +0.125] | no |
| truth_cities | final | length_bucket | 5% | 8/8 | 8 / 7 | +0.046 | [-0.024, +0.115] | [-0.010, +0.101] | no | +0.045 | [-0.021, +0.111] | no |
| truth_cities | final | length_bucket | 10% | 8/8 | 8 / 7 | +0.019 | [-0.035, +0.073] | [-0.024, +0.062] | yes | +0.025 | [-0.024, +0.075] | yes |
| truth_cities | final | named_continent | 0.25% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | final | named_continent | 0.5% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | final | named_continent | 1% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | final | named_continent | 2% | 4/8 | 4 / 4 | +0.111 | [-0.245, +0.467] | [-0.152, +0.374] | no | +0.111 | [-0.152, +0.374] | no |
| truth_cities | final | named_continent | 5% | 7/8 | 7 / 7 | +0.126 | [-0.066, +0.318] | [-0.027, +0.278] | no | +0.126 | [-0.027, +0.278] | no |
| truth_cities | final | named_continent | 10% | 8/8 | 8 / 7 | +0.037 | [-0.149, +0.224] | [-0.112, +0.187] | no | +0.086 | [-0.052, +0.225] | no |
| truth_cities | mid | city_continent | 0.25% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | mid | city_continent | 0.5% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | mid | city_continent | 1% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | mid | city_continent | 2% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | mid | city_continent | 5% | 6/8 | 6 / 6 | +0.018 | [-0.059, +0.094] | [-0.042, +0.077] | yes | +0.018 | [-0.042, +0.077] | yes |
| truth_cities | mid | city_continent | 10% | 7/8 | 7 / 7 | +0.039 | [-0.026, +0.104] | [-0.013, +0.090] | yes | +0.039 | [-0.013, +0.090] | yes |
| truth_cities | mid | length_bucket | 0.25% | 8/8 | 8 / 7 | +0.034 | [-0.108, +0.175] | [-0.080, +0.147] | no | -0.008 | [-0.104, +0.087] | no |
| truth_cities | mid | length_bucket | 0.5% | 8/8 | 8 / 7 | +0.018 | [-0.079, +0.114] | [-0.060, +0.095] | yes | -0.004 | [-0.082, +0.074] | yes |
| truth_cities | mid | length_bucket | 1% | 8/8 | 8 / 7 | +0.017 | [-0.066, +0.100] | [-0.050, +0.083] | yes | -0.010 | [-0.062, +0.043] | yes |
| truth_cities | mid | length_bucket | 2% | 8/8 | 8 / 7 | -0.008 | [-0.086, +0.070] | [-0.070, +0.054] | yes | -0.031 | [-0.084, +0.022] | yes |
| truth_cities | mid | length_bucket | 5% | 8/8 | 8 / 7 | -0.008 | [-0.065, +0.048] | [-0.053, +0.037] | yes | -0.003 | [-0.056, +0.049] | yes |
| truth_cities | mid | length_bucket | 10% | 8/8 | 8 / 7 | +0.005 | [-0.029, +0.040] | [-0.022, +0.033] | yes | -0.001 | [-0.030, +0.028] | yes |
| truth_cities | mid | named_continent | 0.25% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | mid | named_continent | 0.5% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | mid | named_continent | 1% | 0/8 | 0 / 0 | – | – | – | – | – | – | – |
| truth_cities | mid | named_continent | 2% | 3/8 | 3 / 3 | +0.052 | [-0.102, +0.207] | [-0.053, +0.157] | no | +0.052 | [-0.053, +0.157] | no |
| truth_cities | mid | named_continent | 5% | 6/8 | 6 / 6 | +0.076 | [-0.041, +0.192] | [-0.016, +0.167] | no | +0.076 | [-0.016, +0.167] | no |
| truth_cities | mid | named_continent | 10% | 7/8 | 7 / 7 | +0.076 | [+0.026, +0.125] | [+0.036, +0.115] | no | +0.076 | [+0.036, +0.115] | no |

## Control readability per model (retention top / bottom)

| model | dataset | layer | control | k | control top | control bottom | concept top | concept bottom | Δ |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.817 | +0.559 | -0.258 |
| Llama-3.2-3B | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.511 | +0.647 | +0.136 |
| Mistral-7B-v0.3 | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.761 | +0.805 | +0.043 |
| Qwen2.5-0.5B | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.264 | +0.452 | +0.187 |
| Qwen2.5-3B | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.726 | +0.654 | -0.072 |
| Qwen2.5-7B | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.879 | +0.710 | -0.169 |
| Qwen3-4B | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.794 | +0.756 | -0.038 |
| gemma-2-2b | truth_cities | final | city_continent | 0.25% | -0.000 | -0.000 | +0.145 | +0.251 | +0.106 |
| Llama-3.1-8B | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.849 | +0.742 | -0.107 |
| Llama-3.2-3B | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.643 | +0.759 | +0.116 |
| Mistral-7B-v0.3 | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.817 | +0.811 | -0.006 |
| Qwen2.5-0.5B | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.295 | +0.469 | +0.174 |
| Qwen2.5-3B | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.812 | +0.688 | -0.124 |
| Qwen2.5-7B | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.861 | +0.793 | -0.068 |
| Qwen3-4B | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.885 | +0.865 | -0.020 |
| gemma-2-2b | truth_cities | final | city_continent | 0.5% | -0.000 | -0.000 | +0.285 | +0.227 | -0.058 |
| Llama-3.1-8B | truth_cities | final | city_continent | 1% | +0.003 | +0.044 | +0.897 | +0.875 | -0.063 |
| Llama-3.2-3B | truth_cities | final | city_continent | 1% | +0.045 | -0.000 | +0.844 | +0.838 | +0.039 |
| Mistral-7B-v0.3 | truth_cities | final | city_continent | 1% | +0.046 | +0.003 | +0.874 | +0.848 | +0.017 |
| Qwen2.5-0.5B | truth_cities | final | city_continent | 1% | -0.000 | +0.004 | +0.331 | +0.591 | +0.256 |
| Qwen2.5-3B | truth_cities | final | city_continent | 1% | -0.000 | +0.003 | +0.852 | +0.770 | -0.085 |
| Qwen2.5-7B | truth_cities | final | city_continent | 1% | -0.000 | -0.000 | +0.911 | +0.889 | -0.022 |
| Qwen3-4B | truth_cities | final | city_continent | 1% | -0.000 | +0.012 | +0.919 | +0.913 | -0.018 |
| gemma-2-2b | truth_cities | final | city_continent | 1% | -0.000 | +0.030 | +0.715 | +0.299 | -0.446 |
| Llama-3.1-8B | truth_cities | final | city_continent | 2% | +0.133 | +0.148 | +0.907 | +0.921 | -0.001 |
| Llama-3.2-3B | truth_cities | final | city_continent | 2% | +0.273 | +0.117 | +0.919 | +0.899 | +0.136 |
| Mistral-7B-v0.3 | truth_cities | final | city_continent | 2% | +0.396 | +0.053 | +0.933 | +0.911 | +0.322 |
| Qwen2.5-0.5B | truth_cities | final | city_continent | 2% | -0.000 | +0.093 | +0.672 | +0.756 | -0.009 |
| Qwen2.5-3B | truth_cities | final | city_continent | 2% | +0.007 | +0.042 | +0.864 | +0.896 | -0.003 |
| Qwen2.5-7B | truth_cities | final | city_continent | 2% | -0.004 | +0.118 | +0.924 | +0.920 | -0.126 |
| Qwen3-4B | truth_cities | final | city_continent | 2% | +0.015 | +0.015 | +0.966 | +0.958 | -0.008 |
| gemma-2-2b | truth_cities | final | city_continent | 2% | -0.013 | +0.175 | +0.560 | +0.608 | -0.140 |
| Llama-3.1-8B | truth_cities | final | city_continent | 5% | +0.551 | +0.528 | +0.972 | +0.956 | +0.008 |
| Llama-3.2-3B | truth_cities | final | city_continent | 5% | +0.631 | +0.466 | +0.939 | +0.968 | +0.195 |
| Mistral-7B-v0.3 | truth_cities | final | city_continent | 5% | +0.706 | +0.356 | +0.965 | +0.961 | +0.346 |
| Qwen2.5-0.5B | truth_cities | final | city_continent | 5% | +0.127 | +0.346 | +0.826 | +0.852 | -0.193 |
| Qwen2.5-3B | truth_cities | final | city_continent | 5% | +0.306 | +0.236 | +0.926 | +0.946 | +0.089 |
| Qwen2.5-7B | truth_cities | final | city_continent | 5% | +0.258 | +0.394 | +0.940 | +0.954 | -0.122 |
| Qwen3-4B | truth_cities | final | city_continent | 5% | +0.424 | +0.240 | +0.980 | +0.982 | +0.186 |
| gemma-2-2b | truth_cities | final | city_continent | 5% | +0.020 | +0.285 | +0.831 | +0.831 | -0.265 |
| Llama-3.1-8B | truth_cities | final | city_continent | 10% | +0.794 | +0.782 | +0.984 | +0.990 | +0.018 |
| Llama-3.2-3B | truth_cities | final | city_continent | 10% | +0.814 | +0.661 | +0.972 | +0.982 | +0.163 |
| Mistral-7B-v0.3 | truth_cities | final | city_continent | 10% | +0.815 | +0.597 | +0.974 | +0.970 | +0.214 |
| Qwen2.5-0.5B | truth_cities | final | city_continent | 10% | +0.338 | +0.528 | +0.919 | +0.930 | -0.179 |
| Qwen2.5-3B | truth_cities | final | city_continent | 10% | +0.705 | +0.424 | +0.962 | +0.974 | +0.293 |
| Qwen2.5-7B | truth_cities | final | city_continent | 10% | +0.563 | +0.667 | +0.962 | +0.972 | -0.094 |
| Qwen3-4B | truth_cities | final | city_continent | 10% | +0.743 | +0.562 | +0.992 | +0.988 | +0.177 |
| gemma-2-2b | truth_cities | final | city_continent | 10% | +0.332 | +0.541 | +0.923 | +0.884 | -0.247 |
| Llama-3.1-8B | truth_cities | final | length_bucket | 0.25% | +0.322 | +0.353 | +0.817 | +0.559 | -0.289 |
| Llama-3.2-3B | truth_cities | final | length_bucket | 0.25% | +0.422 | +0.409 | +0.511 | +0.647 | +0.149 |
| Mistral-7B-v0.3 | truth_cities | final | length_bucket | 0.25% | +0.375 | +0.293 | +0.761 | +0.805 | +0.125 |
| Qwen2.5-0.5B | truth_cities | final | length_bucket | 0.25% | +0.159 | +0.204 | +0.264 | +0.452 | +0.142 |
| Qwen2.5-3B | truth_cities | final | length_bucket | 0.25% | +0.241 | +0.312 | +0.726 | +0.654 | -0.142 |
| Qwen2.5-7B | truth_cities | final | length_bucket | 0.25% | +0.132 | +0.237 | +0.879 | +0.710 | -0.274 |
| Qwen3-4B | truth_cities | final | length_bucket | 0.25% | +0.192 | +0.377 | +0.794 | +0.756 | -0.224 |
| gemma-2-2b | truth_cities | final | length_bucket | 0.25% | +0.438 | +0.293 | +0.145 | +0.251 | +0.251 |
| Llama-3.1-8B | truth_cities | final | length_bucket | 0.5% | +0.410 | +0.413 | +0.849 | +0.742 | -0.110 |
| Llama-3.2-3B | truth_cities | final | length_bucket | 0.5% | +0.476 | +0.521 | +0.643 | +0.759 | +0.072 |
| Mistral-7B-v0.3 | truth_cities | final | length_bucket | 0.5% | +0.354 | +0.391 | +0.817 | +0.811 | -0.044 |
| Qwen2.5-0.5B | truth_cities | final | length_bucket | 0.5% | +0.210 | +0.300 | +0.295 | +0.469 | +0.084 |
| Qwen2.5-3B | truth_cities | final | length_bucket | 0.5% | +0.439 | +0.406 | +0.812 | +0.688 | -0.090 |
| Qwen2.5-7B | truth_cities | final | length_bucket | 0.5% | +0.214 | +0.197 | +0.861 | +0.793 | -0.051 |
| Qwen3-4B | truth_cities | final | length_bucket | 0.5% | +0.198 | +0.396 | +0.885 | +0.865 | -0.218 |
| gemma-2-2b | truth_cities | final | length_bucket | 0.5% | +0.401 | +0.434 | +0.285 | +0.227 | -0.091 |
| Llama-3.1-8B | truth_cities | final | length_bucket | 1% | +0.539 | +0.442 | +0.897 | +0.875 | +0.076 |
| Llama-3.2-3B | truth_cities | final | length_bucket | 1% | +0.604 | +0.527 | +0.844 | +0.838 | +0.071 |
| Mistral-7B-v0.3 | truth_cities | final | length_bucket | 1% | +0.533 | +0.451 | +0.874 | +0.848 | +0.056 |
| Qwen2.5-0.5B | truth_cities | final | length_bucket | 1% | +0.183 | +0.400 | +0.331 | +0.591 | +0.043 |
| Qwen2.5-3B | truth_cities | final | length_bucket | 1% | +0.483 | +0.399 | +0.852 | +0.770 | +0.002 |
| Qwen2.5-7B | truth_cities | final | length_bucket | 1% | +0.404 | +0.374 | +0.911 | +0.889 | +0.008 |
| Qwen3-4B | truth_cities | final | length_bucket | 1% | +0.537 | +0.453 | +0.919 | +0.913 | +0.077 |
| gemma-2-2b | truth_cities | final | length_bucket | 1% | +0.579 | +0.628 | +0.715 | +0.299 | -0.465 |
| Llama-3.1-8B | truth_cities | final | length_bucket | 2% | +0.704 | +0.502 | +0.907 | +0.921 | +0.216 |
| Llama-3.2-3B | truth_cities | final | length_bucket | 2% | +0.617 | +0.652 | +0.919 | +0.899 | -0.055 |
| Mistral-7B-v0.3 | truth_cities | final | length_bucket | 2% | +0.587 | +0.616 | +0.933 | +0.911 | -0.050 |
| Qwen2.5-0.5B | truth_cities | final | length_bucket | 2% | +0.427 | +0.531 | +0.672 | +0.756 | -0.020 |
| Qwen2.5-3B | truth_cities | final | length_bucket | 2% | +0.503 | +0.470 | +0.864 | +0.896 | +0.066 |
| Qwen2.5-7B | truth_cities | final | length_bucket | 2% | +0.615 | +0.497 | +0.924 | +0.920 | +0.114 |
| Qwen3-4B | truth_cities | final | length_bucket | 2% | +0.617 | +0.517 | +0.966 | +0.958 | +0.091 |
| gemma-2-2b | truth_cities | final | length_bucket | 2% | +0.615 | +0.769 | +0.560 | +0.608 | -0.105 |
| Llama-3.1-8B | truth_cities | final | length_bucket | 5% | +0.789 | +0.748 | +0.972 | +0.956 | +0.025 |
| Llama-3.2-3B | truth_cities | final | length_bucket | 5% | +0.728 | +0.802 | +0.939 | +0.968 | -0.044 |
| Mistral-7B-v0.3 | truth_cities | final | length_bucket | 5% | +0.697 | +0.748 | +0.965 | +0.961 | -0.054 |
| Qwen2.5-0.5B | truth_cities | final | length_bucket | 5% | +0.572 | +0.607 | +0.826 | +0.852 | -0.008 |
| Qwen2.5-3B | truth_cities | final | length_bucket | 5% | +0.658 | +0.507 | +0.926 | +0.946 | +0.171 |
| Qwen2.5-7B | truth_cities | final | length_bucket | 5% | +0.776 | +0.711 | +0.940 | +0.954 | +0.080 |
| Qwen3-4B | truth_cities | final | length_bucket | 5% | +0.678 | +0.533 | +0.980 | +0.982 | +0.146 |
| gemma-2-2b | truth_cities | final | length_bucket | 5% | +0.818 | +0.769 | +0.831 | +0.831 | +0.050 |
| Llama-3.1-8B | truth_cities | final | length_bucket | 10% | +0.846 | +0.808 | +0.984 | +0.990 | +0.044 |
| Llama-3.2-3B | truth_cities | final | length_bucket | 10% | +0.824 | +0.910 | +0.972 | +0.982 | -0.076 |
| Mistral-7B-v0.3 | truth_cities | final | length_bucket | 10% | +0.842 | +0.896 | +0.974 | +0.970 | -0.058 |
| Qwen2.5-0.5B | truth_cities | final | length_bucket | 10% | +0.751 | +0.727 | +0.919 | +0.930 | +0.035 |
| Qwen2.5-3B | truth_cities | final | length_bucket | 10% | +0.722 | +0.661 | +0.962 | +0.974 | +0.072 |
| Qwen2.5-7B | truth_cities | final | length_bucket | 10% | +0.849 | +0.803 | +0.962 | +0.972 | +0.056 |
| Qwen3-4B | truth_cities | final | length_bucket | 10% | +0.783 | +0.674 | +0.992 | +0.988 | +0.105 |
| gemma-2-2b | truth_cities | final | length_bucket | 10% | +0.847 | +0.835 | +0.923 | +0.884 | -0.026 |
| Llama-3.1-8B | truth_cities | final | named_continent | 0.25% | -0.000 | +0.003 | +0.817 | +0.559 | -0.261 |
| Llama-3.2-3B | truth_cities | final | named_continent | 0.25% | -0.000 | -0.000 | +0.511 | +0.647 | +0.136 |
| Mistral-7B-v0.3 | truth_cities | final | named_continent | 0.25% | -0.000 | -0.000 | +0.761 | +0.805 | +0.043 |
| Qwen2.5-0.5B | truth_cities | final | named_continent | 0.25% | -0.000 | -0.000 | +0.264 | +0.452 | +0.187 |
| Qwen2.5-3B | truth_cities | final | named_continent | 0.25% | -0.000 | -0.000 | +0.726 | +0.654 | -0.072 |
| Qwen2.5-7B | truth_cities | final | named_continent | 0.25% | -0.000 | -0.000 | +0.879 | +0.710 | -0.169 |
| Qwen3-4B | truth_cities | final | named_continent | 0.25% | -0.000 | -0.000 | +0.794 | +0.756 | -0.038 |
| gemma-2-2b | truth_cities | final | named_continent | 0.25% | -0.000 | -0.000 | +0.145 | +0.251 | +0.106 |
| Llama-3.1-8B | truth_cities | final | named_continent | 0.5% | +0.015 | +0.017 | +0.849 | +0.742 | -0.110 |
| Llama-3.2-3B | truth_cities | final | named_continent | 0.5% | -0.000 | -0.000 | +0.643 | +0.759 | +0.116 |
| Mistral-7B-v0.3 | truth_cities | final | named_continent | 0.5% | -0.000 | -0.000 | +0.817 | +0.811 | -0.006 |
| Qwen2.5-0.5B | truth_cities | final | named_continent | 0.5% | -0.000 | -0.000 | +0.295 | +0.469 | +0.174 |
| Qwen2.5-3B | truth_cities | final | named_continent | 0.5% | -0.000 | -0.000 | +0.812 | +0.688 | -0.124 |
| Qwen2.5-7B | truth_cities | final | named_continent | 0.5% | -0.000 | -0.000 | +0.861 | +0.793 | -0.068 |
| Qwen3-4B | truth_cities | final | named_continent | 0.5% | -0.000 | -0.000 | +0.885 | +0.865 | -0.020 |
| gemma-2-2b | truth_cities | final | named_continent | 0.5% | -0.000 | -0.000 | +0.285 | +0.227 | -0.058 |
| Llama-3.1-8B | truth_cities | final | named_continent | 1% | +0.058 | +0.055 | +0.897 | +0.875 | -0.019 |
| Llama-3.2-3B | truth_cities | final | named_continent | 1% | +0.038 | -0.000 | +0.844 | +0.838 | +0.032 |
| Mistral-7B-v0.3 | truth_cities | final | named_continent | 1% | +0.065 | +0.017 | +0.874 | +0.848 | +0.022 |
| Qwen2.5-0.5B | truth_cities | final | named_continent | 1% | -0.000 | -0.000 | +0.331 | +0.591 | +0.260 |
| Qwen2.5-3B | truth_cities | final | named_continent | 1% | -0.000 | -0.000 | +0.852 | +0.770 | -0.082 |
| Qwen2.5-7B | truth_cities | final | named_continent | 1% | -0.000 | +0.015 | +0.911 | +0.889 | -0.037 |
| Qwen3-4B | truth_cities | final | named_continent | 1% | -0.000 | -0.010 | +0.919 | +0.913 | +0.004 |
| gemma-2-2b | truth_cities | final | named_continent | 1% | -0.000 | -0.000 | +0.715 | +0.299 | -0.416 |
| Llama-3.1-8B | truth_cities | final | named_continent | 2% | +0.202 | +0.144 | +0.907 | +0.921 | +0.072 |
| Llama-3.2-3B | truth_cities | final | named_continent | 2% | +0.239 | +0.012 | +0.919 | +0.899 | +0.208 |
| Mistral-7B-v0.3 | truth_cities | final | named_continent | 2% | +0.431 | +0.065 | +0.933 | +0.911 | +0.344 |
| Qwen2.5-0.5B | truth_cities | final | named_continent | 2% | -0.000 | +0.027 | +0.672 | +0.756 | +0.056 |
| Qwen2.5-3B | truth_cities | final | named_continent | 2% | +0.023 | +0.011 | +0.864 | +0.896 | +0.043 |
| Qwen2.5-7B | truth_cities | final | named_continent | 2% | +0.012 | +0.188 | +0.924 | +0.920 | -0.180 |
| Qwen3-4B | truth_cities | final | named_continent | 2% | +0.034 | -0.007 | +0.966 | +0.958 | +0.033 |
| gemma-2-2b | truth_cities | final | named_continent | 2% | -0.000 | +0.014 | +0.560 | +0.608 | +0.034 |
| Llama-3.1-8B | truth_cities | final | named_continent | 5% | +0.595 | +0.460 | +0.972 | +0.956 | +0.120 |
| Llama-3.2-3B | truth_cities | final | named_continent | 5% | +0.539 | +0.309 | +0.939 | +0.968 | +0.260 |
| Mistral-7B-v0.3 | truth_cities | final | named_continent | 5% | +0.755 | +0.408 | +0.965 | +0.961 | +0.343 |
| Qwen2.5-0.5B | truth_cities | final | named_continent | 5% | +0.070 | +0.328 | +0.826 | +0.852 | -0.232 |
| Qwen2.5-3B | truth_cities | final | named_continent | 5% | +0.419 | +0.183 | +0.926 | +0.946 | +0.255 |
| Qwen2.5-7B | truth_cities | final | named_continent | 5% | +0.312 | +0.405 | +0.940 | +0.954 | -0.078 |
| Qwen3-4B | truth_cities | final | named_continent | 5% | +0.388 | +0.179 | +0.980 | +0.982 | +0.212 |
| gemma-2-2b | truth_cities | final | named_continent | 5% | -0.022 | +0.029 | +0.831 | +0.831 | -0.051 |
| Llama-3.1-8B | truth_cities | final | named_continent | 10% | +0.789 | +0.720 | +0.984 | +0.990 | +0.075 |
| Llama-3.2-3B | truth_cities | final | named_continent | 10% | +0.723 | +0.542 | +0.972 | +0.982 | +0.191 |
| Mistral-7B-v0.3 | truth_cities | final | named_continent | 10% | +0.848 | +0.707 | +0.974 | +0.970 | +0.137 |
| Qwen2.5-0.5B | truth_cities | final | named_continent | 10% | +0.258 | +0.489 | +0.919 | +0.930 | -0.220 |
| Qwen2.5-3B | truth_cities | final | named_continent | 10% | +0.722 | +0.418 | +0.962 | +0.974 | +0.316 |
| Qwen2.5-7B | truth_cities | final | named_continent | 10% | +0.613 | +0.729 | +0.962 | +0.972 | -0.106 |
| Qwen3-4B | truth_cities | final | named_continent | 10% | +0.670 | +0.454 | +0.992 | +0.988 | +0.212 |
| gemma-2-2b | truth_cities | final | named_continent | 10% | -0.015 | +0.254 | +0.923 | +0.884 | -0.307 |
| Llama-3.1-8B | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.923 | +0.953 | +0.029 |
| Llama-3.2-3B | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.775 | +0.894 | +0.118 |
| Mistral-7B-v0.3 | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.926 | +0.976 | +0.051 |
| Qwen2.5-0.5B | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.613 | +0.569 | -0.045 |
| Qwen2.5-3B | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.758 | +0.631 | -0.128 |
| Qwen2.5-7B | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.868 | +0.809 | -0.059 |
| Qwen3-4B | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.798 | +0.762 | -0.036 |
| gemma-2-2b | truth_cities | mid | city_continent | 0.25% | -0.000 | -0.000 | +0.174 | +0.369 | +0.195 |
| Llama-3.1-8B | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.961 | +0.988 | +0.028 |
| Llama-3.2-3B | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.911 | +0.984 | +0.073 |
| Mistral-7B-v0.3 | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.939 | +0.986 | +0.047 |
| Qwen2.5-0.5B | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.713 | +0.845 | +0.132 |
| Qwen2.5-3B | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.937 | +0.792 | -0.145 |
| Qwen2.5-7B | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.945 | +0.931 | -0.014 |
| Qwen3-4B | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.909 | +0.883 | -0.026 |
| gemma-2-2b | truth_cities | mid | city_continent | 0.5% | -0.000 | -0.000 | +0.250 | +0.589 | +0.339 |
| Llama-3.1-8B | truth_cities | mid | city_continent | 1% | -0.004 | +0.004 | +0.961 | +0.978 | +0.011 |
| Llama-3.2-3B | truth_cities | mid | city_continent | 1% | -0.000 | -0.000 | +0.949 | +0.984 | +0.036 |
| Mistral-7B-v0.3 | truth_cities | mid | city_continent | 1% | -0.000 | +0.019 | +0.971 | +0.996 | +0.007 |
| Qwen2.5-0.5B | truth_cities | mid | city_continent | 1% | -0.000 | -0.000 | +0.745 | +0.845 | +0.100 |
| Qwen2.5-3B | truth_cities | mid | city_continent | 1% | +0.024 | -0.000 | +0.973 | +0.898 | -0.051 |
| Qwen2.5-7B | truth_cities | mid | city_continent | 1% | +0.003 | +0.016 | +0.980 | +0.970 | -0.023 |
| Qwen3-4B | truth_cities | mid | city_continent | 1% | -0.000 | -0.000 | +0.942 | +0.944 | +0.002 |
| gemma-2-2b | truth_cities | mid | city_continent | 1% | -0.000 | -0.000 | +0.517 | +0.775 | +0.259 |
| Llama-3.1-8B | truth_cities | mid | city_continent | 2% | +0.018 | +0.035 | +0.965 | +0.978 | -0.004 |
| Llama-3.2-3B | truth_cities | mid | city_continent | 2% | +0.004 | -0.000 | +0.982 | +0.986 | +0.008 |
| Mistral-7B-v0.3 | truth_cities | mid | city_continent | 2% | +0.012 | +0.053 | +0.988 | +0.998 | -0.031 |
| Qwen2.5-0.5B | truth_cities | mid | city_continent | 2% | +0.011 | -0.000 | +0.802 | +0.890 | +0.098 |
| Qwen2.5-3B | truth_cities | mid | city_continent | 2% | +0.072 | -0.000 | +0.990 | +0.986 | +0.068 |
| Qwen2.5-7B | truth_cities | mid | city_continent | 2% | +0.023 | +0.033 | +0.980 | +0.972 | -0.018 |
| Qwen3-4B | truth_cities | mid | city_continent | 2% | +0.004 | +0.026 | +0.964 | +0.972 | -0.015 |
| gemma-2-2b | truth_cities | mid | city_continent | 2% | -0.000 | -0.000 | +0.737 | +0.886 | +0.148 |
| Llama-3.1-8B | truth_cities | mid | city_continent | 5% | +0.148 | +0.165 | +0.988 | +0.986 | -0.020 |
| Llama-3.2-3B | truth_cities | mid | city_continent | 5% | +0.126 | +0.033 | +0.992 | +0.994 | +0.096 |
| Mistral-7B-v0.3 | truth_cities | mid | city_continent | 5% | +0.184 | +0.249 | +0.994 | +1.000 | -0.060 |
| Qwen2.5-0.5B | truth_cities | mid | city_continent | 5% | +0.022 | +0.011 | +0.862 | +0.917 | +0.066 |
| Qwen2.5-3B | truth_cities | mid | city_continent | 5% | +0.212 | +0.109 | +1.000 | +0.992 | +0.095 |
| Qwen2.5-7B | truth_cities | mid | city_continent | 5% | +0.233 | +0.180 | +0.988 | +0.988 | +0.052 |
| Qwen3-4B | truth_cities | mid | city_continent | 5% | +0.090 | +0.146 | +0.982 | +0.980 | -0.058 |
| gemma-2-2b | truth_cities | mid | city_continent | 5% | -0.000 | -0.000 | +0.877 | +0.847 | -0.030 |
| Llama-3.1-8B | truth_cities | mid | city_continent | 10% | +0.419 | +0.377 | +0.996 | +0.994 | +0.040 |
| Llama-3.2-3B | truth_cities | mid | city_continent | 10% | +0.311 | +0.184 | +0.994 | +0.998 | +0.130 |
| Mistral-7B-v0.3 | truth_cities | mid | city_continent | 10% | +0.452 | +0.486 | +0.998 | +1.000 | -0.032 |
| Qwen2.5-0.5B | truth_cities | mid | city_continent | 10% | +0.178 | +0.072 | +0.958 | +0.968 | +0.116 |
| Qwen2.5-3B | truth_cities | mid | city_continent | 10% | +0.394 | +0.342 | +0.998 | +0.998 | +0.051 |
| Qwen2.5-7B | truth_cities | mid | city_continent | 10% | +0.561 | +0.534 | +0.996 | +0.994 | +0.024 |
| Qwen3-4B | truth_cities | mid | city_continent | 10% | +0.255 | +0.311 | +0.992 | +0.988 | -0.060 |
| gemma-2-2b | truth_cities | mid | city_continent | 10% | +0.005 | +0.032 | +0.860 | +0.894 | +0.007 |
| Llama-3.1-8B | truth_cities | mid | length_bucket | 0.25% | +0.389 | +0.409 | +0.923 | +0.953 | +0.010 |
| Llama-3.2-3B | truth_cities | mid | length_bucket | 0.25% | +0.440 | +0.496 | +0.775 | +0.894 | +0.063 |
| Mistral-7B-v0.3 | truth_cities | mid | length_bucket | 0.25% | +0.367 | +0.530 | +0.926 | +0.976 | -0.112 |
| Qwen2.5-0.5B | truth_cities | mid | length_bucket | 0.25% | +0.233 | -0.045 | +0.613 | +0.569 | +0.233 |
| Qwen2.5-3B | truth_cities | mid | length_bucket | 0.25% | +0.439 | +0.463 | +0.758 | +0.631 | -0.152 |
| Qwen2.5-7B | truth_cities | mid | length_bucket | 0.25% | +0.465 | +0.494 | +0.868 | +0.809 | -0.088 |
| Qwen3-4B | truth_cities | mid | length_bucket | 0.25% | +0.299 | +0.275 | +0.798 | +0.762 | -0.011 |
| gemma-2-2b | truth_cities | mid | length_bucket | 0.25% | +0.504 | +0.371 | +0.174 | +0.369 | +0.328 |
| Llama-3.1-8B | truth_cities | mid | length_bucket | 0.5% | +0.420 | +0.558 | +0.961 | +0.988 | -0.111 |
| Llama-3.2-3B | truth_cities | mid | length_bucket | 0.5% | +0.599 | +0.574 | +0.911 | +0.984 | +0.098 |
| Mistral-7B-v0.3 | truth_cities | mid | length_bucket | 0.5% | +0.568 | +0.593 | +0.939 | +0.986 | +0.023 |
| Qwen2.5-0.5B | truth_cities | mid | length_bucket | 0.5% | +0.361 | +0.322 | +0.713 | +0.845 | +0.170 |
| Qwen2.5-3B | truth_cities | mid | length_bucket | 0.5% | +0.517 | +0.480 | +0.937 | +0.792 | -0.107 |
| Qwen2.5-7B | truth_cities | mid | length_bucket | 0.5% | +0.549 | +0.586 | +0.945 | +0.931 | -0.051 |
| Qwen3-4B | truth_cities | mid | length_bucket | 0.5% | +0.407 | +0.431 | +0.909 | +0.883 | -0.050 |
| gemma-2-2b | truth_cities | mid | length_bucket | 0.5% | +0.467 | +0.638 | +0.250 | +0.589 | +0.168 |
| Llama-3.1-8B | truth_cities | mid | length_bucket | 1% | +0.594 | +0.671 | +0.961 | +0.978 | -0.058 |
| Llama-3.2-3B | truth_cities | mid | length_bucket | 1% | +0.680 | +0.643 | +0.949 | +0.984 | +0.072 |
| Mistral-7B-v0.3 | truth_cities | mid | length_bucket | 1% | +0.663 | +0.710 | +0.971 | +0.996 | -0.021 |
| Qwen2.5-0.5B | truth_cities | mid | length_bucket | 1% | +0.477 | +0.477 | +0.745 | +0.845 | +0.100 |
| Qwen2.5-3B | truth_cities | mid | length_bucket | 1% | +0.539 | +0.553 | +0.973 | +0.898 | -0.088 |
| Qwen2.5-7B | truth_cities | mid | length_bucket | 1% | +0.625 | +0.683 | +0.980 | +0.970 | -0.068 |
| Qwen3-4B | truth_cities | mid | length_bucket | 1% | +0.528 | +0.534 | +0.942 | +0.944 | -0.003 |
| gemma-2-2b | truth_cities | mid | length_bucket | 1% | +0.646 | +0.704 | +0.517 | +0.775 | +0.200 |
| Llama-3.1-8B | truth_cities | mid | length_bucket | 2% | +0.690 | +0.783 | +0.965 | +0.978 | -0.079 |
| Llama-3.2-3B | truth_cities | mid | length_bucket | 2% | +0.657 | +0.724 | +0.982 | +0.986 | -0.063 |
| Mistral-7B-v0.3 | truth_cities | mid | length_bucket | 2% | +0.685 | +0.786 | +0.988 | +0.998 | -0.091 |
| Qwen2.5-0.5B | truth_cities | mid | length_bucket | 2% | +0.615 | +0.642 | +0.802 | +0.890 | +0.060 |
| Qwen2.5-3B | truth_cities | mid | length_bucket | 2% | +0.574 | +0.648 | +0.990 | +0.986 | -0.077 |
| Qwen2.5-7B | truth_cities | mid | length_bucket | 2% | +0.707 | +0.749 | +0.980 | +0.972 | -0.050 |
| Qwen3-4B | truth_cities | mid | length_bucket | 2% | +0.671 | +0.596 | +0.964 | +0.972 | +0.083 |
| gemma-2-2b | truth_cities | mid | length_bucket | 2% | +0.700 | +0.696 | +0.737 | +0.886 | +0.152 |
| Llama-3.1-8B | truth_cities | mid | length_bucket | 5% | +0.809 | +0.814 | +0.988 | +0.986 | -0.008 |
| Llama-3.2-3B | truth_cities | mid | length_bucket | 5% | +0.786 | +0.813 | +0.992 | +0.994 | -0.026 |
| Mistral-7B-v0.3 | truth_cities | mid | length_bucket | 5% | +0.775 | +0.867 | +0.994 | +1.000 | -0.087 |
| Qwen2.5-0.5B | truth_cities | mid | length_bucket | 5% | +0.776 | +0.698 | +0.862 | +0.917 | +0.133 |
| Qwen2.5-3B | truth_cities | mid | length_bucket | 5% | +0.769 | +0.810 | +1.000 | +0.992 | -0.049 |
| Qwen2.5-7B | truth_cities | mid | length_bucket | 5% | +0.839 | +0.868 | +0.988 | +0.988 | -0.029 |
| Qwen3-4B | truth_cities | mid | length_bucket | 5% | +0.800 | +0.757 | +0.982 | +0.980 | +0.041 |
| gemma-2-2b | truth_cities | mid | length_bucket | 5% | +0.713 | +0.725 | +0.877 | +0.847 | -0.042 |
| Llama-3.1-8B | truth_cities | mid | length_bucket | 10% | +0.873 | +0.854 | +0.996 | +0.994 | +0.018 |
| Llama-3.2-3B | truth_cities | mid | length_bucket | 10% | +0.841 | +0.875 | +0.994 | +0.998 | -0.029 |
| Mistral-7B-v0.3 | truth_cities | mid | length_bucket | 10% | +0.859 | +0.861 | +0.998 | +1.000 | -0.001 |
| Qwen2.5-0.5B | truth_cities | mid | length_bucket | 10% | +0.833 | +0.788 | +0.958 | +0.968 | +0.055 |
| Qwen2.5-3B | truth_cities | mid | length_bucket | 10% | +0.810 | +0.867 | +0.998 | +0.998 | -0.057 |
| Qwen2.5-7B | truth_cities | mid | length_bucket | 10% | +0.902 | +0.929 | +0.996 | +0.994 | -0.028 |
| Qwen3-4B | truth_cities | mid | length_bucket | 10% | +0.876 | +0.838 | +0.992 | +0.988 | +0.034 |
| gemma-2-2b | truth_cities | mid | length_bucket | 10% | +0.792 | +0.775 | +0.860 | +0.894 | +0.051 |
| Llama-3.1-8B | truth_cities | mid | named_continent | 0.25% | -0.000 | -0.000 | +0.923 | +0.953 | +0.029 |
| Llama-3.2-3B | truth_cities | mid | named_continent | 0.25% | -0.000 | -0.000 | +0.775 | +0.894 | +0.118 |
| Mistral-7B-v0.3 | truth_cities | mid | named_continent | 0.25% | -0.000 | -0.000 | +0.926 | +0.976 | +0.051 |
| Qwen2.5-0.5B | truth_cities | mid | named_continent | 0.25% | -0.000 | -0.000 | +0.613 | +0.569 | -0.045 |
| Qwen2.5-3B | truth_cities | mid | named_continent | 0.25% | +0.006 | -0.000 | +0.758 | +0.631 | -0.122 |
| Qwen2.5-7B | truth_cities | mid | named_continent | 0.25% | -0.000 | -0.000 | +0.868 | +0.809 | -0.059 |
| Qwen3-4B | truth_cities | mid | named_continent | 0.25% | -0.000 | -0.000 | +0.798 | +0.762 | -0.036 |
| gemma-2-2b | truth_cities | mid | named_continent | 0.25% | -0.000 | -0.000 | +0.174 | +0.369 | +0.195 |
| Llama-3.1-8B | truth_cities | mid | named_continent | 0.5% | -0.000 | -0.000 | +0.961 | +0.988 | +0.028 |
| Llama-3.2-3B | truth_cities | mid | named_continent | 0.5% | -0.000 | -0.000 | +0.911 | +0.984 | +0.073 |
| Mistral-7B-v0.3 | truth_cities | mid | named_continent | 0.5% | -0.000 | -0.000 | +0.939 | +0.986 | +0.047 |
| Qwen2.5-0.5B | truth_cities | mid | named_continent | 0.5% | -0.000 | -0.000 | +0.713 | +0.845 | +0.132 |
| Qwen2.5-3B | truth_cities | mid | named_continent | 0.5% | +0.006 | -0.000 | +0.937 | +0.792 | -0.139 |
| Qwen2.5-7B | truth_cities | mid | named_continent | 0.5% | -0.000 | +0.011 | +0.945 | +0.931 | -0.025 |
| Qwen3-4B | truth_cities | mid | named_continent | 0.5% | -0.000 | -0.000 | +0.909 | +0.883 | -0.026 |
| gemma-2-2b | truth_cities | mid | named_continent | 0.5% | -0.000 | -0.000 | +0.250 | +0.589 | +0.339 |
| Llama-3.1-8B | truth_cities | mid | named_continent | 1% | -0.000 | -0.000 | +0.961 | +0.978 | +0.018 |
| Llama-3.2-3B | truth_cities | mid | named_continent | 1% | +0.003 | +0.019 | +0.949 | +0.984 | +0.019 |
| Mistral-7B-v0.3 | truth_cities | mid | named_continent | 1% | +0.046 | +0.041 | +0.971 | +0.996 | +0.031 |
| Qwen2.5-0.5B | truth_cities | mid | named_continent | 1% | -0.000 | -0.000 | +0.745 | +0.845 | +0.100 |
| Qwen2.5-3B | truth_cities | mid | named_continent | 1% | +0.075 | -0.000 | +0.973 | +0.898 | -0.000 |
| Qwen2.5-7B | truth_cities | mid | named_continent | 1% | +0.003 | +0.050 | +0.980 | +0.970 | -0.057 |
| Qwen3-4B | truth_cities | mid | named_continent | 1% | +0.006 | +0.011 | +0.942 | +0.944 | -0.004 |
| gemma-2-2b | truth_cities | mid | named_continent | 1% | -0.000 | -0.000 | +0.517 | +0.775 | +0.259 |
| Llama-3.1-8B | truth_cities | mid | named_continent | 2% | +0.020 | +0.028 | +0.965 | +0.978 | +0.005 |
| Llama-3.2-3B | truth_cities | mid | named_continent | 2% | +0.019 | +0.053 | +0.982 | +0.986 | -0.029 |
| Mistral-7B-v0.3 | truth_cities | mid | named_continent | 2% | +0.217 | +0.109 | +0.988 | +0.998 | +0.118 |
| Qwen2.5-0.5B | truth_cities | mid | named_continent | 2% | -0.000 | -0.000 | +0.802 | +0.890 | +0.087 |
| Qwen2.5-3B | truth_cities | mid | named_continent | 2% | +0.134 | +0.086 | +0.990 | +0.986 | +0.044 |
| Qwen2.5-7B | truth_cities | mid | named_continent | 2% | +0.131 | +0.128 | +0.980 | +0.972 | -0.005 |
| Qwen3-4B | truth_cities | mid | named_continent | 2% | +0.084 | +0.053 | +0.964 | +0.972 | +0.039 |
| gemma-2-2b | truth_cities | mid | named_continent | 2% | -0.000 | -0.000 | +0.737 | +0.886 | +0.148 |
| Llama-3.1-8B | truth_cities | mid | named_continent | 5% | +0.334 | +0.252 | +0.988 | +0.986 | +0.080 |
| Llama-3.2-3B | truth_cities | mid | named_continent | 5% | +0.213 | +0.175 | +0.992 | +0.994 | +0.041 |
| Mistral-7B-v0.3 | truth_cities | mid | named_continent | 5% | +0.568 | +0.285 | +0.994 | +1.000 | +0.288 |
| Qwen2.5-0.5B | truth_cities | mid | named_continent | 5% | +0.047 | +0.006 | +0.862 | +0.917 | +0.096 |
| Qwen2.5-3B | truth_cities | mid | named_continent | 5% | +0.224 | +0.242 | +1.000 | +0.992 | -0.026 |
| Qwen2.5-7B | truth_cities | mid | named_continent | 5% | +0.447 | +0.383 | +0.988 | +0.988 | +0.064 |
| Qwen3-4B | truth_cities | mid | named_continent | 5% | +0.267 | +0.258 | +0.982 | +0.980 | +0.006 |
| gemma-2-2b | truth_cities | mid | named_continent | 5% | -0.000 | -0.000 | +0.877 | +0.847 | -0.030 |
| Llama-3.1-8B | truth_cities | mid | named_continent | 10% | +0.629 | +0.547 | +0.996 | +0.994 | +0.080 |
| Llama-3.2-3B | truth_cities | mid | named_continent | 10% | +0.488 | +0.421 | +0.994 | +0.998 | +0.070 |
| Mistral-7B-v0.3 | truth_cities | mid | named_continent | 10% | +0.764 | +0.581 | +0.998 | +1.000 | +0.184 |
| Qwen2.5-0.5B | truth_cities | mid | named_continent | 10% | +0.246 | +0.199 | +0.958 | +0.968 | +0.058 |
| Qwen2.5-3B | truth_cities | mid | named_continent | 10% | +0.493 | +0.415 | +0.998 | +0.998 | +0.078 |
| Qwen2.5-7B | truth_cities | mid | named_continent | 10% | +0.747 | +0.733 | +0.996 | +0.994 | +0.012 |
| Qwen3-4B | truth_cities | mid | named_continent | 10% | +0.528 | +0.477 | +0.992 | +0.988 | +0.047 |
| gemma-2-2b | truth_cities | mid | named_continent | 10% | +0.010 | -0.000 | +0.860 | +0.894 | +0.044 |
