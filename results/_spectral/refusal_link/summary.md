# Refusal: dual geometry and link with refusal words

Positive gap = further toward the head than the null. Published dual geometry: vocab gap > 0 and activation gap < 0. Link > 0: the refusal direction points toward the refusal words more than toward random word contrasts.

## Token rule: published (primary)

| quantity | variant, layer | models | n | mean | 95% CI | p | 90% CI | p TOST |
|---|---|---|---|---|---|---|---|---|
| vocabulary gap | – | all | 8 | -0.058 | [-0.070, -0.046] | <0.001 | [-0.068, -0.048] | 0.001 |
| vocabulary gap | – | w/o gemma | 7 | -0.061 | [-0.073, -0.049] | <0.001 | [-0.070, -0.052] | 0.002 |
| activation gap | plain, final | all | 8 | -0.020 | [-0.049, +0.009] | 0.150 | [-0.043, +0.003] | <0.001 |
| activation gap | plain, final | w/o gemma | 7 | -0.023 | [-0.057, +0.011] | 0.153 | [-0.050, +0.004] | 0.003 |
| link (word null) | plain, final | all | 8 | +0.019 | [+0.009, +0.029] | 0.002 | [+0.011, +0.027] | <0.001 |
| link (word null) | plain, final | w/o gemma | 7 | +0.020 | [+0.009, +0.032] | 0.005 | [+0.011, +0.029] | <0.001 |
| activation gap | plain, mid | all | 8 | -0.026 | [-0.055, +0.003] | 0.072 | [-0.049, -0.003] | 0.001 |
| activation gap | plain, mid | w/o gemma | 7 | -0.034 | [-0.060, -0.008] | 0.019 | [-0.055, -0.013] | 0.002 |
| link (word null) | plain, mid | all | 8 | +0.027 | [+0.012, +0.043] | 0.004 | [+0.015, +0.039] | <0.001 |
| link (word null) | plain, mid | w/o gemma | 7 | +0.028 | [+0.009, +0.046] | 0.010 | [+0.013, +0.042] | <0.001 |
| activation gap | prompted, final | all | 8 | -0.005 | [-0.028, +0.019] | 0.662 | [-0.024, +0.014] | <0.001 |
| activation gap | prompted, final | w/o gemma | 7 | -0.010 | [-0.034, +0.015] | 0.369 | [-0.029, +0.010] | <0.001 |
| link (word null) | prompted, final | all | 8 | +0.054 | [+0.028, +0.080] | 0.002 | [+0.033, +0.075] | 0.020 |
| link (word null) | prompted, final | w/o gemma | 7 | +0.059 | [+0.030, +0.087] | 0.002 | [+0.036, +0.081] | 0.047 |
| activation gap | prompted, mid | all | 8 | -0.003 | [-0.011, +0.005] | 0.374 | [-0.010, +0.003] | <0.001 |
| activation gap | prompted, mid | w/o gemma | 7 | -0.005 | [-0.013, +0.002] | 0.120 | [-0.011, 0.000] | <0.001 |
| link (word null) | prompted, mid | all | 8 | +0.006 | [-0.007, +0.019] | 0.310 | [-0.004, +0.016] | <0.001 |
| link (word null) | prompted, mid | w/o gemma | 7 | +0.006 | [-0.009, +0.021] | 0.336 | [-0.006, +0.018] | <0.001 |

## Token rule: space_first

| quantity | variant, layer | models | n | mean | 95% CI | p | 90% CI | p TOST |
|---|---|---|---|---|---|---|---|---|
| vocabulary gap | – | all | 8 | -0.072 | [-0.083, -0.061] | <0.001 | [-0.081, -0.063] | 0.038 |
| vocabulary gap | – | w/o gemma | 7 | -0.074 | [-0.086, -0.062] | <0.001 | [-0.084, -0.064] | 0.083 |
| activation gap | plain, final | all | 8 | -0.020 | [-0.049, +0.009] | 0.150 | [-0.043, +0.003] | <0.001 |
| activation gap | plain, final | w/o gemma | 7 | -0.023 | [-0.057, +0.011] | 0.153 | [-0.050, +0.004] | 0.003 |
| link (word null) | plain, final | all | 8 | +0.019 | [+0.011, +0.027] | <0.001 | [+0.013, +0.026] | <0.001 |
| link (word null) | plain, final | w/o gemma | 7 | +0.021 | [+0.012, +0.030] | 0.001 | [+0.013, +0.028] | <0.001 |
| activation gap | plain, mid | all | 8 | -0.026 | [-0.055, +0.003] | 0.072 | [-0.049, -0.003] | 0.001 |
| activation gap | plain, mid | w/o gemma | 7 | -0.034 | [-0.060, -0.008] | 0.019 | [-0.055, -0.013] | 0.002 |
| link (word null) | plain, mid | all | 8 | +0.031 | [+0.021, +0.042] | <0.001 | [+0.023, +0.039] | <0.001 |
| link (word null) | plain, mid | w/o gemma | 7 | +0.033 | [+0.021, +0.044] | <0.001 | [+0.023, +0.042] | <0.001 |
| activation gap | prompted, final | all | 8 | -0.005 | [-0.028, +0.019] | 0.662 | [-0.024, +0.014] | <0.001 |
| activation gap | prompted, final | w/o gemma | 7 | -0.010 | [-0.034, +0.015] | 0.369 | [-0.029, +0.010] | <0.001 |
| link (word null) | prompted, final | all | 8 | +0.057 | [+0.026, +0.088] | 0.003 | [+0.032, +0.082] | 0.049 |
| link (word null) | prompted, final | w/o gemma | 7 | +0.063 | [+0.030, +0.096] | 0.003 | [+0.037, +0.089] | 0.102 |
| activation gap | prompted, mid | all | 8 | -0.003 | [-0.011, +0.005] | 0.374 | [-0.010, +0.003] | <0.001 |
| activation gap | prompted, mid | w/o gemma | 7 | -0.005 | [-0.013, +0.002] | 0.120 | [-0.011, 0.000] | <0.001 |
| link (word null) | prompted, mid | all | 8 | +0.003 | [-0.013, +0.019] | 0.694 | [-0.010, +0.016] | <0.001 |
| link (word null) | prompted, mid | w/o gemma | 7 | +0.003 | [-0.016, +0.023] | 0.670 | [-0.012, +0.019] | <0.001 |

**Primary 1, dual geometry (plain, final layer, published rule): NOT DUAL**
**Primary 2, link (prompted, final layer, published rule): LINKED**
