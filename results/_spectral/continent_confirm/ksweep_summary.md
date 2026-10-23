# EXPLORATORY: continent control, window-size sweep

Not pre-registered. At the pre-registered k = 0.5% the continent labels were unreadable at both ends, so Delta there is not a specificity statistic. A cell counts as READABLE when the control's retention exceeds 0.1 in the top or the bottom window; only readable, usable cells enter the across-model test.

| dataset | layer | control | k | readable models | n (all / w/o gemma) | mean Δ all | 95% CI | 90% CI | equiv. | mean Δ w/o gemma | 90% CI | equiv. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth_cities_negated | final | city_continent | 10% | 8/8 | 8 / 7 | +0.011 | [-0.107, +0.129] | [-0.083, +0.106] | no | +0.038 | [-0.056, +0.132] | no |
| truth_cities_negated | final | length_bucket | 10% | 8/8 | 8 / 7 | -0.017 | [-0.075, +0.041] | [-0.063, +0.029] | yes | -0.022 | [-0.076, +0.031] | yes |
| truth_cities_negated | final | named_continent | 10% | 8/8 | 8 / 7 | +0.020 | [-0.156, +0.197] | [-0.121, +0.162] | no | +0.069 | [-0.059, +0.197] | no |
| truth_cities_negated | mid | city_continent | 10% | 7/8 | 7 / 7 | +0.080 | [-0.011, +0.172] | [+0.007, +0.153] | no | +0.080 | [+0.007, +0.153] | no |
| truth_cities_negated | mid | length_bucket | 10% | 8/8 | 8 / 7 | -0.032 | [-0.062, -0.003] | [-0.056, -0.009] | yes | -0.035 | [-0.062, -0.008] | yes |
| truth_cities_negated | mid | named_continent | 10% | 8/8 | 8 / 7 | +0.029 | [-0.032, +0.089] | [-0.020, +0.077] | yes | +0.016 | [-0.034, +0.066] | yes |

## Control readability per model (retention top / bottom)

| model | dataset | layer | control | k | control top | control bottom | concept top | concept bottom | Δ |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | truth_cities_negated | final | city_continent | 10% | +0.891 | +0.907 | +0.988 | +0.996 | -0.008 |
| Llama-3.2-3B | truth_cities_negated | final | city_continent | 10% | +0.910 | +0.844 | +0.994 | +0.992 | +0.064 |
| Mistral-7B-v0.3 | truth_cities_negated | final | city_continent | 10% | +0.896 | +0.786 | +0.998 | +0.992 | +0.104 |
| Qwen2.5-0.5B | truth_cities_negated | final | city_continent | 10% | +0.255 | +0.502 | +0.876 | +0.914 | -0.209 |
| Qwen2.5-3B | truth_cities_negated | final | city_continent | 10% | +0.547 | +0.395 | +0.953 | +0.966 | +0.166 |
| Qwen2.5-7B | truth_cities_negated | final | city_continent | 10% | +0.551 | +0.554 | +0.976 | +0.978 | -0.002 |
| Qwen3-4B | truth_cities_negated | final | city_continent | 10% | +0.727 | +0.568 | +1.000 | +0.992 | +0.150 |
| gemma-2-2b | truth_cities_negated | final | city_continent | 10% | +0.375 | +0.510 | +0.958 | +0.916 | -0.177 |
| Llama-3.1-8B | truth_cities_negated | final | length_bucket | 10% | +0.816 | +0.852 | +0.988 | +0.996 | -0.029 |
| Llama-3.2-3B | truth_cities_negated | final | length_bucket | 10% | +0.793 | +0.959 | +0.994 | +0.992 | -0.168 |
| Mistral-7B-v0.3 | truth_cities_negated | final | length_bucket | 10% | +0.735 | +0.757 | +0.998 | +0.992 | -0.028 |
| Qwen2.5-0.5B | truth_cities_negated | final | length_bucket | 10% | +0.687 | +0.753 | +0.876 | +0.914 | -0.028 |
| Qwen2.5-3B | truth_cities_negated | final | length_bucket | 10% | +0.756 | +0.769 | +0.953 | +0.966 | +0.001 |
| Qwen2.5-7B | truth_cities_negated | final | length_bucket | 10% | +0.823 | +0.778 | +0.976 | +0.978 | +0.047 |
| Qwen3-4B | truth_cities_negated | final | length_bucket | 10% | +0.793 | +0.737 | +1.000 | +0.992 | +0.048 |
| gemma-2-2b | truth_cities_negated | final | length_bucket | 10% | +0.946 | +0.882 | +0.958 | +0.916 | +0.022 |
| Llama-3.1-8B | truth_cities_negated | final | named_continent | 10% | +0.764 | +0.711 | +0.988 | +0.996 | +0.061 |
| Llama-3.2-3B | truth_cities_negated | final | named_continent | 10% | +0.753 | +0.658 | +0.994 | +0.992 | +0.093 |
| Mistral-7B-v0.3 | truth_cities_negated | final | named_continent | 10% | +0.730 | +0.654 | +0.998 | +0.992 | +0.070 |
| Qwen2.5-0.5B | truth_cities_negated | final | named_continent | 10% | +0.205 | +0.487 | +0.876 | +0.914 | -0.244 |
| Qwen2.5-3B | truth_cities_negated | final | named_continent | 10% | +0.773 | +0.555 | +0.953 | +0.966 | +0.232 |
| Qwen2.5-7B | truth_cities_negated | final | named_continent | 10% | +0.699 | +0.723 | +0.976 | +0.978 | -0.023 |
| Qwen3-4B | truth_cities_negated | final | named_continent | 10% | +0.640 | +0.343 | +1.000 | +0.992 | +0.289 |
| gemma-2-2b | truth_cities_negated | final | named_continent | 10% | -0.032 | +0.242 | +0.958 | +0.916 | -0.316 |
| Llama-3.1-8B | truth_cities_negated | mid | city_continent | 10% | +0.647 | +0.514 | +0.998 | +1.000 | +0.135 |
| Llama-3.2-3B | truth_cities_negated | mid | city_continent | 10% | +0.485 | +0.397 | +1.000 | +1.000 | +0.089 |
| Mistral-7B-v0.3 | truth_cities_negated | mid | city_continent | 10% | +0.809 | +0.552 | +1.000 | +1.000 | +0.258 |
| Qwen2.5-0.5B | truth_cities_negated | mid | city_continent | 10% | +0.144 | +0.059 | +0.970 | +0.972 | +0.087 |
| Qwen2.5-3B | truth_cities_negated | mid | city_continent | 10% | +0.436 | +0.393 | +0.996 | +1.000 | +0.048 |
| Qwen2.5-7B | truth_cities_negated | mid | city_continent | 10% | +0.545 | +0.575 | +1.000 | +0.996 | -0.034 |
| Qwen3-4B | truth_cities_negated | mid | city_continent | 10% | +0.409 | +0.432 | +0.996 | +0.998 | -0.021 |
| gemma-2-2b | truth_cities_negated | mid | city_continent | 10% | +0.018 | +0.004 | +0.983 | +0.992 | +0.022 |
| Llama-3.1-8B | truth_cities_negated | mid | length_bucket | 10% | +0.827 | +0.860 | +0.998 | +1.000 | -0.032 |
| Llama-3.2-3B | truth_cities_negated | mid | length_bucket | 10% | +0.804 | +0.855 | +1.000 | +1.000 | -0.050 |
| Mistral-7B-v0.3 | truth_cities_negated | mid | length_bucket | 10% | +0.824 | +0.902 | +1.000 | +1.000 | -0.078 |
| Qwen2.5-0.5B | truth_cities_negated | mid | length_bucket | 10% | +0.856 | +0.853 | +0.970 | +0.972 | +0.005 |
| Qwen2.5-3B | truth_cities_negated | mid | length_bucket | 10% | +0.835 | +0.821 | +0.996 | +1.000 | +0.018 |
| Qwen2.5-7B | truth_cities_negated | mid | length_bucket | 10% | +0.842 | +0.914 | +1.000 | +0.996 | -0.076 |
| Qwen3-4B | truth_cities_negated | mid | length_bucket | 10% | +0.863 | +0.898 | +0.996 | +0.998 | -0.033 |
| gemma-2-2b | truth_cities_negated | mid | length_bucket | 10% | +0.765 | +0.787 | +0.983 | +0.992 | -0.013 |
| Llama-3.1-8B | truth_cities_negated | mid | named_continent | 10% | +0.661 | +0.675 | +0.998 | +1.000 | -0.012 |
| Llama-3.2-3B | truth_cities_negated | mid | named_continent | 10% | +0.525 | +0.600 | +1.000 | +1.000 | -0.075 |
| Mistral-7B-v0.3 | truth_cities_negated | mid | named_continent | 10% | +0.702 | +0.680 | +1.000 | +1.000 | +0.022 |
| Qwen2.5-0.5B | truth_cities_negated | mid | named_continent | 10% | +0.297 | +0.213 | +0.970 | +0.972 | +0.086 |
| Qwen2.5-3B | truth_cities_negated | mid | named_continent | 10% | +0.540 | +0.481 | +0.996 | +1.000 | +0.063 |
| Qwen2.5-7B | truth_cities_negated | mid | named_continent | 10% | +0.765 | +0.822 | +1.000 | +0.996 | -0.061 |
| Qwen3-4B | truth_cities_negated | mid | named_continent | 10% | +0.598 | +0.510 | +0.996 | +0.998 | +0.090 |
| gemma-2-2b | truth_cities_negated | mid | named_continent | 10% | +0.107 | -0.000 | +0.983 | +0.992 | +0.116 |
