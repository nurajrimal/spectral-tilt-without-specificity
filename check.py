import sys, json, glob, os
sys.path.insert(0, 'src')
import torch
from linear_rep_geometry.behavioral import load_concept

print("=== 1. gemma vs qwen: is the data sane? ===")
for m in ['gemma-2-2b', 'Qwen2.5-3B']:
    p = f'results/{m}/concept_refusal.pt'
    if not os.path.exists(p):
        print(f"  {m}: missing {p}")
        continue
    c = load_concept(p)
    L = c['n_layers'] // 2
    D = c['diffs'][('last_token', L)]
    print(f"  {m:16s} n={c['n_pairs']} d={c['d']} layers={c['n_layers']}")
    print(f"      diff norm mean {float(D.norm(dim=1).mean()):.4f}  "
          f"std {float(D.norm(dim=1).std()):.4f}  "
          f"nan={bool(torch.isnan(D).any())}  inf={bool(torch.isinf(D).any())}")

print()
print("=== 2. word-level control present? ===")
for m in ['gemma-2-2b', 'Qwen2.5-3B']:
    n = len(glob.glob(f'results/{m}/concepts/concept_word_*.pt'))
    print(f"  {m:16s} {n} word concepts on disk")

print()
print("=== 3. Phase 4: is the scalar Gini the right statistic? ===")
for m in ['Qwen2.5-3B', 'gemma-2-2b']:
    p = f'results/{m}/phase4_full_vocab.json'
    if not os.path.exists(p):
        print(f"  {m}: no phase4 file")
        continue
    d = json.load(open(p))
    print(f"  {m}:")
    for c in ['refusal', 'truth_cities', 'sycophancy']:
        if c in d.get('concepts', {}):
            s = d['concepts'][c]['summary']
            print(f"      {c:14s} PC1 {s['pc1_var_mean']:.0%}  "
                  f"PR {s['participation_ratio_mean']:.2f}  "
                  f"rank-one: {s['treat_as_rank_one']}")
