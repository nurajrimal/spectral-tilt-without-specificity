# Two-tail report

Status: **PRE-REGISTERED (frozen 2026-10-09T23:12:48.104495+00:00  sha256 e492f247ba2e)**

Equivalence bound (SESOI): ±0.1 retention units. Primary k: loud 0.1, quiet 0.005.


## Primary tests and the Holm family

| models | tail | layer | quantity | n | mean | 95% CI | p | p (Holm) | 90% CI | p TOST | equiv. | p non-sup. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | loud | final | delta **(primary)** | 8 | -0.040 | [-0.073, -0.007] | 0.025 | 0.203 | [-0.066, -0.013] | 0.002 | yes | <0.001 |
| all | loud | final | gap | 8 | +0.030 | [-0.005, +0.065] | 0.079 | 0.476 | [+0.002, +0.058] | 0.001 | yes | 0.001 |
| all | loud | mid | delta | 8 | -0.034 | [-0.063, -0.004] | 0.030 | 0.209 | [-0.057, -0.010] | <0.001 | yes | <0.001 |
| all | loud | mid | gap | 8 | +0.024 | [-0.029, +0.077] | 0.314 | 1.000 | [-0.018, +0.067] | 0.006 | yes | 0.006 |
| all | quiet | mid | delta **(primary)** | 8 | -0.003 | [-0.068, +0.062] | 0.920 | 1.000 | [-0.055, +0.049] | 0.005 | yes | 0.004 |
| all | quiet | mid | gap | 8 | +0.002 | [-0.050, +0.053] | 0.932 | 1.000 | [-0.039, +0.043] | 0.001 | yes | 0.001 |
| all | quiet | final | delta | 8 | -0.022 | [-0.069, +0.026] | 0.314 | 1.000 | [-0.060, +0.016] | 0.003 | yes | <0.001 |
| all | quiet | final | gap | 8 | -0.009 | [-0.040, +0.022] | 0.531 | 1.000 | [-0.034, +0.016] | <0.001 | yes | <0.001 |
| without gemma | loud | final | delta **(primary)** | 7 | -0.034 | [-0.071, +0.002] | 0.062 | 0.431 | [-0.063, -0.005] | 0.002 | yes | <0.001 |
| without gemma | loud | final | gap | 7 | +0.024 | [-0.014, +0.061] | 0.171 | 1.000 | [-0.006, +0.053] | 0.001 | yes | 0.001 |
| without gemma | loud | mid | delta | 7 | -0.036 | [-0.071, -0.002] | 0.043 | 0.343 | [-0.064, -0.009] | 0.002 | yes | <0.001 |
| without gemma | loud | mid | gap | 7 | +0.004 | [-0.022, +0.030] | 0.728 | 1.000 | [-0.017, +0.024] | <0.001 | yes | <0.001 |
| without gemma | quiet | mid | delta **(primary)** | 7 | -0.009 | [-0.084, +0.066] | 0.774 | 1.000 | [-0.069, +0.051] | 0.013 | yes | 0.006 |
| without gemma | quiet | mid | gap | 7 | -0.011 | [-0.060, +0.039] | 0.608 | 1.000 | [-0.050, +0.028] | 0.002 | yes | <0.001 |
| without gemma | quiet | final | delta | 7 | -0.030 | [-0.082, +0.022] | 0.205 | 1.000 | [-0.071, +0.011] | 0.008 | yes | <0.001 |
| without gemma | quiet | final | gap | 7 | -0.010 | [-0.047, +0.027] | 0.542 | 1.000 | [-0.039, +0.020] | <0.001 | yes | <0.001 |

## Window-size sweep (Δ, all models / without gemma)

| tail | layer | k=0.0025 | k=0.005 | k=0.01 | k=0.02 | k=0.05 | k=0.1 |
|---|---|---|---|---|---|---|---|
| loud | final | -0.000 (p 0.997) / -0.003 (p 0.898) | -0.005 (p 0.868) / -0.017 (p 0.583) | +0.012 (p 0.593) / +0.014 (p 0.584) | -0.004 (p 0.904) / +0.003 (p 0.930) | -0.038 (p 0.042) / -0.029 (p 0.091) | -0.040 (p 0.025) / -0.034 (p 0.062) |
| loud | mid | +0.012 (p 0.195) / +0.013 (p 0.230) | +0.031 (p 0.039) / +0.026 (p 0.091) | +0.036 (p 0.330) / +0.017 (p 0.622) | +0.044 (p 0.348) / +0.009 (p 0.778) | +0.008 (p 0.716) / -0.011 (p 0.361) | -0.034 (p 0.030) / -0.036 (p 0.043) |
| quiet | final | -0.018 (p 0.472) / -0.016 (p 0.576) | -0.022 (p 0.314) / -0.030 (p 0.205) | -0.035 (p 0.178) / -0.024 (p 0.355) | +0.008 (p 0.638) / +0.014 (p 0.437) | +0.004 (p 0.685) / +0.009 (p 0.277) | +0.014 (p 0.185) / +0.018 (p 0.141) |
| quiet | mid | -0.005 (p 0.868) / -0.002 (p 0.956) | -0.003 (p 0.920) / -0.009 (p 0.774) | +0.013 (p 0.682) / -0.003 (p 0.908) | -0.011 (p 0.752) / -0.027 (p 0.427) | -0.021 (p 0.385) / -0.031 (p 0.217) | -0.020 (p 0.263) / -0.028 (p 0.147) |

The primary k is fixed in the pre-registration; the sweep is robustness only.

## Loud tail against each control (final layer, primary k)

| control | models | n | models where POS beats control | mean Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| token_id_random | all | 8 | 2 | -0.040 | [-0.073, -0.007] | 0.025 |
| token_id_random | without gemma | 7 | 2 | -0.034 | [-0.071, +0.002] | 0.062 |
| freq_bucket | all | 8 | 3 | -0.007 | [-0.039, +0.025] | 0.609 |
| freq_bucket | without gemma | 7 | 2 | -0.012 | [-0.048, +0.024] | 0.440 |
| length_bucket | all | 8 | 3 | -0.015 | [-0.049, +0.019] | 0.331 |
| length_bucket | without gemma | 7 | 2 | -0.024 | [-0.054, +0.005] | 0.090 |

## How strongly length predicts the label

AUC 0.5 = length says nothing about the label.

| dataset | pairs | AUC chars | AUC tokens | pos longer | neg longer | tied |
|---|---|---|---|---|---|---|
| sycophancy | 512 | 0.500 | 0.500 | 0% | 0% | 100% |
| sycophancy_nlp | 500 | 0.500 | 0.500 | 0% | 0% | 100% |
| sycophancy_philpapers | 500 | 0.500 | 0.500 | 0% | 0% | 100% |
| sycophancy_political | 500 | 0.500 | 0.500 | 0% | 0% | 100% |
| truth_cities | 512 | 0.487 | 0.498 | 40% | 41% | 19% |
| truth_cities_negated | 512 | 0.513 | 0.502 | 41% | 40% | 19% |
| truth_conjunction | 512 | 0.489 | 0.515 | 46% | 50% | 4% |
| truth_counterfact | 512 | 0.495 | 0.500 | 36% | 40% | 24% |
| truth_disjunction | 512 | 0.480 | 0.515 | 44% | 51% | 5% |
| truth_magnitude | 512 | 0.640 | 0.500 | 87% | 13% | 0% |
| truth_translation | 354 | 0.504 | 0.506 | 50% | 50% | 0% |

## Robustness analyses

| analysis | tail | layer | quantity | models | n | mean | 95% CI | p | p TOST |
|---|---|---|---|---|---|---|---|---|---|
| length balanced | quiet | mid | delta | all | 8 | +0.013 | [-0.075, +0.102] | 0.733 | 0.027 |
| length balanced | quiet | mid | delta | without gemma | 7 | +0.008 | [-0.096, +0.113] | 0.853 | 0.038 |
| length balanced | quiet | mid | gap | all | 8 | +0.006 | [-0.056, +0.068] | 0.829 | 0.004 |
| length balanced | quiet | mid | gap | without gemma | 7 | -0.007 | [-0.071, +0.056] | 0.789 | 0.006 |
| length balanced | quiet | final | delta | all | 8 | -0.010 | [-0.058, +0.039] | 0.643 | 0.002 |
| length balanced | quiet | final | delta | without gemma | 7 | -0.021 | [-0.070, +0.027] | 0.328 | 0.004 |
| length balanced | quiet | final | gap | all | 8 | -0.005 | [-0.035, +0.026] | 0.736 | <0.001 |
| length balanced | quiet | final | gap | without gemma | 7 | -0.005 | [-0.042, +0.032] | 0.756 | <0.001 |
| length residualised | quiet | mid | delta | all | 8 | -0.007 | [-0.067, +0.053] | 0.792 | 0.004 |
| length residualised | quiet | mid | delta | without gemma | 7 | -0.009 | [-0.081, +0.062] | 0.758 | 0.010 |
| length residualised | quiet | mid | gap | all | 8 | -0.001 | [-0.044, +0.042] | 0.957 | <0.001 |
| length residualised | quiet | mid | gap | without gemma | 7 | -0.010 | [-0.055, +0.036] | 0.619 | 0.001 |
| length residualised | quiet | final | delta | all | 8 | -0.014 | [-0.065, +0.037] | 0.538 | 0.003 |
| length residualised | quiet | final | delta | without gemma | 7 | -0.025 | [-0.077, +0.027] | 0.282 | 0.006 |
| length residualised | quiet | final | gap | all | 8 | -0.001 | [-0.036, +0.034] | 0.956 | <0.001 |
| length residualised | quiet | final | gap | without gemma | 7 | -0.005 | [-0.045, +0.036] | 0.777 | <0.001 |
| MLP probe (quiet) | quiet | mid | delta | all | 4 | -0.034 | [-0.086, +0.018] | 0.131 | 0.014 |
| MLP probe (quiet) | quiet | mid | delta | without gemma | 4 | -0.034 | [-0.086, +0.018] | 0.131 | 0.014 |
| MLP probe (quiet) | quiet | mid | gap | all | 4 | -0.012 | [-0.030, +0.006] | 0.120 | <0.001 |
| MLP probe (quiet) | quiet | mid | gap | without gemma | 4 | -0.012 | [-0.030, +0.006] | 0.120 | <0.001 |
| MLP probe (quiet) | quiet | final | delta | all | 4 | -0.009 | [-0.068, +0.050] | 0.656 | 0.008 |
| MLP probe (quiet) | quiet | final | delta | without gemma | 4 | -0.009 | [-0.068, +0.050] | 0.656 | 0.008 |
| MLP probe (quiet) | quiet | final | gap | all | 4 | +0.000 | [-0.043, +0.043] | 0.987 | 0.003 |
| MLP probe (quiet) | quiet | final | gap | without gemma | 4 | +0.000 | [-0.043, +0.043] | 0.987 | 0.003 |
| MLP probe (loud) | loud | final | delta | all | 4 | -0.000 | [-0.022, +0.022] | 0.997 | <0.001 |
| MLP probe (loud) | loud | final | delta | without gemma | 4 | -0.000 | [-0.022, +0.022] | 0.997 | <0.001 |
| MLP probe (loud) | loud | final | gap | all | 4 | -0.003 | [-0.051, +0.044] | 0.845 | 0.004 |
| MLP probe (loud) | loud | final | gap | without gemma | 4 | -0.003 | [-0.051, +0.044] | 0.845 | 0.004 |
| full-vocabulary basis (loud) | loud | final | delta | all | 8 | -0.010 | [-0.061, +0.040] | 0.636 | 0.002 |
| full-vocabulary basis (loud) | loud | final | delta | without gemma | 7 | -0.014 | [-0.073, +0.045] | 0.583 | 0.006 |
| full-vocabulary basis (loud) | loud | final | gap | all | 8 | -0.019 | [-0.061, +0.024] | 0.333 | 0.001 |
| full-vocabulary basis (loud) | loud | final | gap | without gemma | 7 | -0.028 | [-0.071, +0.015] | 0.159 | 0.003 |

## Placement versus capacity

60 (model, concept) cells. Placement puts energy in the tail (gap < 0) in 30; of those, the tail also holds more of the concept than the head (capacity > 0) in 15. Spearman ρ = -0.39 (p = 0.002; cells within a model are not independent, so read ρ descriptively).

## Sentences selected by the write-up rules

- **\LoudOpening**: Part of speech prefers the head no more than a random lexical partition does
- **\LoudVerdict**: The control is preferred more than part of speech in the all-model version only, so we claim no more than that part of speech is not preferred.
- **\LoudNonSupSentence**: The one-sided test rejects any syntax-specific advantage as large as $+0.10$
- **\QuietOpening**: Concepts prefer the tail no more than surface length does
- **\QuietEquivalenceSentence**: The 90\% interval lies inside the pre-registered bound of $\pm0.10$ in both versions, so the quiet-tail specificity effect is not just undetected but smaller than the smallest effect we said would matter.
- **\QuietAbstractEq**: and the effect lies inside a pre-registered equivalence bound of $\pm0.10$ retention units
- **\HolmSentence**: After Holm correction, with all eight models, no member is significant; without gemma-2-2b, no member is significant.
- **\KSweepSentence**: The mean $\Delta$ is significantly positive in 1 of 48 cells (loud mid at $k=0.5\%$), against about 1.2 expected by chance alone at two-sided $\alpha = 0.05$. These are reported but not interpreted, because $k$ was fixed in advance; at the pre-registered $k$ the conclusion is as stated above.
- **\LengthSentence**: Both stay inside the $\pm0.10$ equivalence bound, so the length control's correlation with the label does not drive the quiet-tail result.
- **\MLPSentence**: so the conclusion is not specific to linear read-out.
- **\ControlsOpening**: Part of speech also fails to beat type frequency and word length
- **\PlacementSentence**: Placement and capacity are only weakly related. Against the structural null, the direction's energy sits in the tail in 30 of 60 cells, and in only 15 of those is the concept also read better from the tail than from the head. Where a direction's energy sits is a poor guide to where its concept can be read.
