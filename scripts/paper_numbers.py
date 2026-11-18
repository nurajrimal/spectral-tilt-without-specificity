#!/usr/bin/env python3
"""Print the paper's quoted single numbers from the committed result files.

Each line gives the value as printed in the paper, the value in the file, and
the file and field it comes from, so a reader can check a number without
re-running anything. Exits non-zero if a value does not round to what the
paper prints.

    python scripts/paper_numbers.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (paper section, what, value printed in the paper, decimals, file, field path)
NUMBERS = [
    ("5.1", "word-level mean Gini, language basis", -0.353, 3,
     "results/_spectral/word_specificity/language_final/Qwen2.5-0.5B.json",
     "summary.all.mean_gini"),
    ("5.1", "condition number, language basis (x10^3)", 4.19, 2,
     "results/_spectral/word_specificity/language_final/Qwen2.5-0.5B.json",
     "condition_number", 1e-3),
    ("5.1", "isotropic baseline, language basis", -0.161, 3,
     "results/_spectral/word_specificity/language_final/Qwen2.5-0.5B.json",
     "isotropic_gini_mean"),
    ("5.1", "word-level mean Gini, full vocabulary", -0.398, 3,
     "results/Qwen2.5-0.5B/phase0.json",
     "word_level.baseline_check.mean_gini"),
    ("5.1", "condition number, full vocabulary (x10^3)", 2.30, 2,
     "results/Qwen2.5-0.5B/phase0.json", "full_vocab.cond", 1e-3),
    ("5.1", "isotropic baseline, full vocabulary", -0.145, 3,
     "results/Qwen2.5-0.5B/phase0.json", "full_vocab.random_gini_mean"),
    ("5.1", "extraction vs published routine, cosine (3pSg-Ved, mean pooling)",
     0.999, 3, "results/Qwen2.5-0.5B/phase0.json",
     "word_level.verification.best_cosine"),
    ("6.2", "continent vs truth, Cramer's V (512 truth_cities pairs)", 0.049, 3,
     "results/_spectral/continent_control/Qwen2.5-0.5B__truth_cities.json",
     "leakage.named_continent.cramers_v"),
    ("6.2", "continent vs truth, chi-square p", 0.48, 2,
     "results/_spectral/continent_control/Qwen2.5-0.5B__truth_cities.json",
     "leakage.named_continent.p"),
    ("6.2", "truth predicted from continent, accuracy", 0.514, 3,
     "results/_spectral/continent_control/Qwen2.5-0.5B__truth_cities.json",
     "leakage.named_continent.truth_acc_from_control"),
]


def get(path, field):
    obj = json.loads((ROOT / path).read_text())
    for key in field.split("."):
        obj = obj[key]
    return float(obj)


def main():
    bad = 0
    for row in NUMBERS:
        sec, what, printed, dp, path, field = row[:6]
        scale = row[6] if len(row) > 6 else 1.0
        try:
            value = get(path, field) * scale
        except (FileNotFoundError, KeyError) as e:
            print(f"MISSING  §{sec} {what}: {path} :: {field} ({e})")
            bad += 1
            continue
        ok = round(value, dp) == round(printed, dp)
        bad += not ok
        print(f"{'ok      ' if ok else 'MISMATCH'} §{sec} {what}: paper {printed} "
              f"file {value:.{dp + 3}f}\n         {path} :: {field}")
    # The continent statistics describe the dataset, not a model: they must
    # be identical in every model's file.
    ps = {round(get(str(p.relative_to(ROOT)), "leakage.named_continent.p"), 12)
          for p in (ROOT / "results/_spectral/continent_control").glob("*__truth_cities*.json")}
    print(f"{'ok      ' if len(ps) == 1 else 'MISMATCH'} continent p identical across "
          f"{len(list((ROOT / 'results/_spectral/continent_control').glob('*__truth_cities*.json')))} files")
    bad += len(ps) != 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
