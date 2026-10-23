#!/usr/bin/env python3
"""
Deep Analysis of RAID Results — Publication-ready figures and statistics.

Runs entirely from saved .pt files (no GPU needed).
1. Re-computes WCA with optimal λ (0.01) from saved covariances
2. Per-concept-group breakdown (morphological vs semantic vs language)
3. Bootstrap confidence intervals on Δ(WCA−Naive)
4. Anisotropy correlation (condition number of Σ vs WCA advantage)

Usage: python scripts/deep_analysis.py results/
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
import glob
import json
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import linear_rep_geometry as lrg

sns.set_theme(context="paper", style="white", palette="colorblind", font="sans-serif", font_scale=1.4)

# ──────────────────────────────────────────────────────────
# Concept Groups
# ──────────────────────────────────────────────────────────
CONCEPT_GROUPS = {
    'Verb Morphology': ['[verb - Ved]', '[verb - Ving]', '[verb - 3pSg]', '[verb - V + er]',
                        '[verb - V + able]', '[verb - V + ment]', '[verb - V + tion]',
                        '[Ving - Ved]', '[Ving - 3pSg]', '[3pSg - Ved]'],
    'Adj Morphology':  ['[adj - comparative]', '[adj - superlative]', '[adj - un + adj]', '[adj - adj + ly]'],
    'Noun Morphology': ['[noun - plural]', '[pronoun - possessive]'],
    'Language Pairs':  ['[English - French]', '[French - German]', '[French - Spanish]', '[German - Spanish]'],
    'Semantic':        ['[male - female]', '[country - capital]', '[thing - color]', '[thing - part]',
                        '[small - big]', '[lower - upper]', '[frequent - infrequent]'],
}

GROUP_COLORS = {
    'Verb Morphology': '#e74c3c',
    'Adj Morphology':  '#3498db',
    'Noun Morphology': '#2ecc71',
    'Language Pairs':  '#9b59b6',
    'Semantic':        '#f39c12',
}

LANG_PAIR_LABELS = {
    'en-fr': 'EN→FR', 'es-de': 'ES→DE', 'fr-de': 'FR→DE', 'fr-es': 'FR→ES'
}


def get_group(name):
    for g, members in CONCEPT_GROUPS.items():
        if name in members:
            return g
    return 'Other'


def cosine_sim(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    denom = (torch.norm(a) * torch.norm(b)).clamp(min=1e-10)
    return (torch.dot(a, b) / denom).item()


# ──────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────
def load_pair_data(pair_dir):
    """Load all vectors and matrices from a completed pair directory."""
    data = {}
    
    # Load covariances
    for src_name, tgt_name in [("cov_src.pt", "cov_tgt.pt"), ("cov_en.pt", "cov_fr.pt")]:
        sp = os.path.join(pair_dir, src_name)
        tp = os.path.join(pair_dir, tgt_name)
        if os.path.exists(sp) and os.path.exists(tp):
            data['cov_src'] = torch.load(sp, map_location='cpu', weights_only=True).float()
            data['cov_tgt'] = torch.load(tp, map_location='cpu', weights_only=True).float()
            break
    
    # Load activations
    for src_name, tgt_name in [("X_src.pt", "X_tgt.pt"), ("X_en.pt", "X_fr.pt")]:
        sp = os.path.join(pair_dir, src_name)
        tp = os.path.join(pair_dir, tgt_name)
        if os.path.exists(sp) and os.path.exists(tp):
            data['X_src'] = torch.load(sp, map_location='cpu', weights_only=True).float()
            data['X_tgt'] = torch.load(tp, map_location='cpu', weights_only=True).float()
            break
    
    # Load concept vectors
    data['raw'] = {}
    data['raid'] = {}
    data['naive'] = {}
    for f in glob.glob(os.path.join(pair_dir, "v_raw_*.pt")):
        name = os.path.basename(f).replace("v_raw_", "").replace(".pt", "")
        data['raw'][name] = torch.load(f, map_location='cpu', weights_only=True).float()
        
        rf = os.path.join(pair_dir, f"v_raid_{name}.pt")
        if os.path.exists(rf):
            data['raid'][name] = torch.load(rf, map_location='cpu', weights_only=True).float()
        
        nf = os.path.join(pair_dir, f"v_naive_{name}.pt")
        if os.path.exists(nf):
            data['naive'][name] = torch.load(nf, map_location='cpu', weights_only=True).float()
    
    return data


def recompute_wca(data, reg_lambda=0.01):
    """Recompute WCA transport with a different λ from saved covariances."""
    if 'cov_src' not in data or 'X_src' not in data:
        return {}
    
    psi_src, _ = lrg.get_whitening_transform(data['cov_src'], reg_lambda=reg_lambda)
    psi_tgt, sqrt_tgt = lrg.get_whitening_transform(data['cov_tgt'], reg_lambda=reg_lambda)
    Q = lrg.solve_whitened_procrustes(data['X_src'], data['X_tgt'], psi_src, psi_tgt)
    
    recomputed = {}
    for name, v_raw in data['raw'].items():
        v_raid = lrg.transport_concept_vector(v_raw, psi_src, sqrt_tgt, Q)
        recomputed[name] = v_raid.cpu()
    
    return recomputed


# ──────────────────────────────────────────────────────────
# Analysis Functions
# ──────────────────────────────────────────────────────────
def compute_deltas(data, recomputed_raid=None):
    """Compute WCA−Naive deltas per concept."""
    raid = recomputed_raid if recomputed_raid else data['raid']
    deltas = {}
    for name in data['raw']:
        if name not in raid or name not in data['naive']:
            continue
        sim_wca = cosine_sim(data['raw'][name], raid[name])
        sim_naive = cosine_sim(data['raw'][name], data['naive'][name])
        deltas[name] = {
            'wca': sim_wca,
            'naive': sim_naive,
            'delta': sim_wca - sim_naive,
            'group': get_group(name)
        }
    return deltas


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    values = np.array(values)
    if len(values) == 0:
        return 0, 0, 0
    boot_means = np.array([np.mean(np.random.choice(values, size=len(values), replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return np.mean(values), lo, hi


def compute_anisotropy(cov_matrix):
    """Compute condition number as measure of anisotropy."""
    eigenvalues = torch.linalg.eigvalsh(cov_matrix.float())
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) < 2:
        return 1.0
    return (eigenvalues.max() / eigenvalues.min()).item()


# ──────────────────────────────────────────────────────────
# Main Analysis
# ──────────────────────────────────────────────────────────
def main(results_root):
    output_dir = os.path.join(results_root, "_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"  RAID Deep Analysis")
    print(f"  Results: {results_root}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*70}\n")
    
    # ── Collect all data ──
    all_data = {}  # {(model, pair): data_dict}
    models = sorted([d for d in os.listdir(results_root) 
                     if os.path.isdir(os.path.join(results_root, d)) and not d.startswith('_')])
    
    for model in models:
        mdir = os.path.join(results_root, model)
        for pair in ['en-fr', 'es-de', 'fr-de', 'fr-es']:
            pdir = os.path.join(mdir, pair)
            if os.path.exists(os.path.join(pdir, "stats.txt")):
                print(f"  Loading {model}/{pair}...")
                data = load_pair_data(pdir)
                if data['raw']:
                    all_data[(model, pair)] = data
    
    print(f"\n  Loaded {len(all_data)} model×pair combinations\n")
    
    # ══════════════════════════════════════════════════════
    # TASK 1: Re-compute with λ=0.01
    # ══════════════════════════════════════════════════════
    print("[1/4] Re-computing WCA with λ=0.01...")
    
    lambdas_to_test = [0.001, 0.01, 0.1]
    results_by_lambda = {lam: [] for lam in lambdas_to_test}
    
    for (model, pair), data in sorted(all_data.items()):
        for lam in lambdas_to_test:
            try:
                recomputed = recompute_wca(data, reg_lambda=lam)
                deltas = compute_deltas(data, recomputed)
                mean_delta = np.mean([d['delta'] for d in deltas.values()])
                mean_wca = np.mean([d['wca'] for d in deltas.values()])
                mean_naive = np.mean([d['naive'] for d in deltas.values()])
                results_by_lambda[lam].append({
                    'model': model, 'pair': pair,
                    'mean_wca': mean_wca, 'mean_naive': mean_naive,
                    'mean_delta': mean_delta,
                    'deltas': deltas
                })
            except Exception as e:
                print(f"    WARN: {model}/{pair} λ={lam} failed: {e}")
    
    # Summary table
    print(f"\n  {'λ':<8} {'WCA Wins':>10} {'Mean Δ':>10}")
    print(f"  {'-'*30}")
    for lam in lambdas_to_test:
        wins = sum(1 for r in results_by_lambda[lam] if r['mean_delta'] > 0)
        total = len(results_by_lambda[lam])
        mean_d = np.mean([r['mean_delta'] for r in results_by_lambda[lam]])
        print(f"  {lam:<8} {wins}/{total:>3} ({100*wins/total:.0f}%)  {mean_d:>+.4f}")
    
    # ── Plot: λ comparison bar chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(lambdas_to_test))
    win_rates = []
    mean_deltas = []
    for lam in lambdas_to_test:
        wins = sum(1 for r in results_by_lambda[lam] if r['mean_delta'] > 0)
        total = len(results_by_lambda[lam])
        win_rates.append(wins / total * 100)
        mean_deltas.append(np.mean([r['mean_delta'] for r in results_by_lambda[lam]]))
    
    bars = ax.bar(x, win_rates, color=['#e74c3c', '#2ecc71', '#3498db'], alpha=0.85, edgecolor='white', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'λ={l}' for l in lambdas_to_test], fontsize=12)
    ax.set_ylabel('WCA Win Rate (%)', fontsize=13)
    ax.set_title('WCA Win Rate by Regularization Strength', fontsize=15, fontweight='bold')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Parity')
    for i, (wr, md) in enumerate(zip(win_rates, mean_deltas)):
        ax.text(i, wr + 1.5, f'{wr:.0f}%\n(Δ={md:+.4f})', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(win_rates) + 15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lambda_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: lambda_comparison.png")
    
    # ══════════════════════════════════════════════════════
    # TASK 2: Per-concept-group breakdown
    # ══════════════════════════════════════════════════════
    print("\n[2/4] Per-concept-group analysis...")
    
    # Use λ=0.01 results
    group_deltas = defaultdict(list)
    for r in results_by_lambda[0.01]:
        for name, d in r['deltas'].items():
            group_deltas[d['group']].append(d['delta'])
    
    # Also compute with default λ=0.001 for comparison
    group_deltas_default = defaultdict(list)
    for r in results_by_lambda[0.001]:
        for name, d in r['deltas'].items():
            group_deltas_default[d['group']].append(d['delta'])
    
    groups_ordered = ['Verb Morphology', 'Adj Morphology', 'Noun Morphology', 'Language Pairs', 'Semantic']
    
    print(f"\n  {'Group':<20} {'λ=0.001 Δ':>12} {'λ=0.01 Δ':>12} {'N':>5}")
    print(f"  {'-'*52}")
    for g in groups_ordered:
        d_def = np.mean(group_deltas_default.get(g, [0]))
        d_opt = np.mean(group_deltas.get(g, [0]))
        n = len(group_deltas.get(g, []))
        marker = '✓' if d_opt > 0 else ' '
        print(f"  {g:<20} {d_def:>+12.4f} {d_opt:>+12.4f} {n:>5} {marker}")
    
    # ── Plot: Per-group comparison ──
    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(groups_ordered))
    bar_h = 0.35
    
    vals_default = [np.mean(group_deltas_default.get(g, [0])) for g in groups_ordered]
    vals_optimal = [np.mean(group_deltas.get(g, [0])) for g in groups_ordered]
    colors = [GROUP_COLORS.get(g, '#95a5a6') for g in groups_ordered]
    
    ax.barh(y_pos - bar_h/2, vals_default, bar_h, label='λ=0.001 (default)',
            color=colors, alpha=0.4, edgecolor='white', linewidth=1, hatch='///')
    ax.barh(y_pos + bar_h/2, vals_optimal, bar_h, label='λ=0.01 (optimal)',
            color=colors, alpha=0.85, edgecolor='white', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups_ordered, fontsize=11)
    ax.set_xlabel('Mean Δ(WCA − Naive)', fontsize=12)
    ax.set_title('WCA Advantage by Concept Group', fontsize=15, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.legend(fontsize=11, loc='lower right')
    ax.invert_yaxis()
    
    # Annotate values
    for i, (vd, vo) in enumerate(zip(vals_default, vals_optimal)):
        ax.text(max(vd, vo) + 0.001, i, f'{vo:+.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "concept_group_breakdown.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: concept_group_breakdown.png")
    
    # ══════════════════════════════════════════════════════
    # TASK 3: Bootstrap confidence intervals
    # ══════════════════════════════════════════════════════
    print("\n[3/4] Bootstrap confidence intervals...")
    
    # Overall CI
    all_deltas_001 = [r['mean_delta'] for r in results_by_lambda[0.001]]
    all_deltas_01 = [r['mean_delta'] for r in results_by_lambda[0.01]]
    all_deltas_1 = [r['mean_delta'] for r in results_by_lambda[0.1]]
    
    for lam, deltas in [(0.001, all_deltas_001), (0.01, all_deltas_01), (0.1, all_deltas_1)]:
        mean, lo, hi = bootstrap_ci(deltas)
        sig = "✓ Significant" if (lo > 0 or hi < 0) else "✗ Not significant"
        print(f"  λ={lam:<6} Mean Δ = {mean:+.4f}  95% CI: [{lo:+.4f}, {hi:+.4f}]  {sig}")
    
    # Per language pair CI (λ=0.01)
    print(f"\n  Per-pair CIs (λ=0.01):")
    pair_cis = {}
    for pair in ['en-fr', 'es-de', 'fr-de', 'fr-es']:
        pair_deltas = [r['mean_delta'] for r in results_by_lambda[0.01] if r['pair'] == pair]
        mean, lo, hi = bootstrap_ci(pair_deltas)
        sig = "✓" if (lo > 0 or hi < 0) else "✗"
        pair_cis[pair] = (mean, lo, hi)
        print(f"    {pair}: Δ = {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")
    
    # Per-group CI (λ=0.01)
    print(f"\n  Per-group CIs (λ=0.01):")
    group_cis = {}
    for g in groups_ordered:
        vals = group_deltas.get(g, [])
        if vals:
            mean, lo, hi = bootstrap_ci(vals)
            sig = "✓" if (lo > 0 or hi < 0) else "✗"
            group_cis[g] = (mean, lo, hi)
            print(f"    {g:<20}: Δ = {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")
    
    # ── Plot: Forest plot of CIs ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: Per-pair CIs
    pairs = list(pair_cis.keys())
    means = [pair_cis[p][0] for p in pairs]
    los = [pair_cis[p][1] for p in pairs]
    his = [pair_cis[p][2] for p in pairs]
    errs = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]
    
    y = np.arange(len(pairs))
    colors_pair = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    ax1.errorbar(means, y, xerr=errs, fmt='o', capsize=6, capthick=2, linewidth=2,
                 markersize=8, color='#2c3e50', ecolor='#7f8c8d')
    for i, (m, p) in enumerate(zip(means, pairs)):
        c = '#2ecc71' if m > 0 else '#e74c3c'
        ax1.scatter([m], [i], color=c, s=100, zorder=5, edgecolors='white', linewidth=1)
    ax1.axvline(0, color='black', linewidth=1, linestyle='--')
    ax1.set_yticks(y)
    ax1.set_yticklabels([LANG_PAIR_LABELS.get(p, p) for p in pairs], fontsize=12)
    ax1.set_xlabel('Δ(WCA − Naive)', fontsize=12)
    ax1.set_title('Per Language Pair (λ=0.01)', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    # Panel 2: Per-group CIs
    groups_with_ci = [g for g in groups_ordered if g in group_cis]
    means_g = [group_cis[g][0] for g in groups_with_ci]
    los_g = [group_cis[g][1] for g in groups_with_ci]
    his_g = [group_cis[g][2] for g in groups_with_ci]
    errs_g = [[m - l for m, l in zip(means_g, los_g)], [h - m for m, h in zip(means_g, his_g)]]
    
    y_g = np.arange(len(groups_with_ci))
    ax2.errorbar(means_g, y_g, xerr=errs_g, fmt='o', capsize=6, capthick=2, linewidth=2,
                 markersize=8, color='#2c3e50', ecolor='#7f8c8d')
    for i, (m, g) in enumerate(zip(means_g, groups_with_ci)):
        c = GROUP_COLORS.get(g, '#95a5a6')
        ax2.scatter([m], [i], color=c, s=100, zorder=5, edgecolors='white', linewidth=1)
    ax2.axvline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_yticks(y_g)
    ax2.set_yticklabels(groups_with_ci, fontsize=11)
    ax2.set_xlabel('Δ(WCA − Naive)', fontsize=12)
    ax2.set_title('Per Concept Group (λ=0.01)', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.suptitle('Bootstrap 95% Confidence Intervals for WCA Advantage', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bootstrap_ci.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bootstrap_ci.png")
    
    # ══════════════════════════════════════════════════════
    # TASK 4: Anisotropy correlation
    # ══════════════════════════════════════════════════════
    print("\n[4/4] Anisotropy correlation...")
    
    aniso_data = []
    for (model, pair), data in sorted(all_data.items()):
        if 'cov_src' not in data:
            continue
        
        cond = compute_anisotropy(data['cov_src'])
        # Get WCA advantage at λ=0.01
        match = [r for r in results_by_lambda[0.01] if r['model'] == model and r['pair'] == pair]
        if match:
            aniso_data.append({
                'model': model, 'pair': pair,
                'condition_number': cond,
                'log_cond': np.log10(cond) if cond > 0 else 0,
                'wca_delta': match[0]['mean_delta'],
                'mean_wca': match[0]['mean_wca'],
                'mean_naive': match[0]['mean_naive'],
            })
    
    if aniso_data:
        log_conds = [d['log_cond'] for d in aniso_data]
        wca_deltas = [d['wca_delta'] for d in aniso_data]
        
        # Correlation
        r_val, p_val = sp_stats.pearsonr(log_conds, wca_deltas)
        print(f"  Pearson r = {r_val:.4f}, p = {p_val:.4e}")
        print(f"  {'✓ Significant' if p_val < 0.05 else '✗ Not significant'} (α=0.05)")
        
        # ── Plot: Scatter ──
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by model family
        family_colors = {
            'Qwen2.5': '#3498db', 'Qwen3': '#2ecc71', 'Llama': '#e74c3c',
            'Mistral': '#9b59b6', 'gemma': '#f39c12', 'SmolLM': '#1abc9c',
            'jetmoe': '#e67e22', 'OLMoE': '#95a5a6'
        }
        
        for d in aniso_data:
            fam = next((f for f in family_colors if f.lower() in d['model'].lower()), 'Other')
            c = family_colors.get(fam, '#95a5a6')
            marker = {'en-fr': 'o', 'es-de': 's', 'fr-de': '^', 'fr-es': 'D'}.get(d['pair'], 'o')
            ax.scatter(d['log_cond'], d['wca_delta'], c=c, marker=marker, s=80,
                      edgecolors='white', linewidth=0.5, alpha=0.8, zorder=3)
        
        # Regression line
        z = np.polyfit(log_conds, wca_deltas, 1)
        x_line = np.linspace(min(log_conds), max(log_conds), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5, linewidth=1.5,
                label=f'r={r_val:.3f}, p={p_val:.2e}')
        
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
        ax.set_xlabel('log₁₀(Condition Number of Σ_src)', fontsize=13)
        ax.set_ylabel('Δ(WCA − Naive)', fontsize=13)
        ax.set_title('Embedding Anisotropy vs. WCA Advantage (λ=0.01)', fontsize=15, fontweight='bold')
        
        # Legends
        for fam, c in family_colors.items():
            ax.plot([], [], 'o', color=c, label=fam, markersize=8)
        for pair, marker in [('en-fr', 'o'), ('es-de', 's'), ('fr-de', '^'), ('fr-es', 'D')]:
            ax.plot([], [], marker, color='gray', label=LANG_PAIR_LABELS[pair], markersize=8)
        ax.legend(fontsize=8, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "anisotropy_correlation.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: anisotropy_correlation.png")
    
    # ══════════════════════════════════════════════════════
    # SUMMARY DASHBOARD
    # ══════════════════════════════════════════════════════
    print("\n[+] Generating summary dashboard...")
    
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    
    # Panel 1: λ comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(lambdas_to_test)), win_rates,
                   color=['#e74c3c', '#2ecc71', '#3498db'], alpha=0.85, edgecolor='white')
    ax1.set_xticks(range(len(lambdas_to_test)))
    ax1.set_xticklabels([f'λ={l}' for l in lambdas_to_test])
    ax1.set_ylabel('WCA Win Rate (%)')
    ax1.set_title('Effect of Regularization', fontsize=12, fontweight='bold')
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
    for i, wr in enumerate(win_rates):
        ax1.text(i, wr + 1, f'{wr:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Panel 2: Per-group bars
    ax2 = fig.add_subplot(gs[0, 1])
    y_g2 = np.arange(len(groups_ordered))
    vals_g = [np.mean(group_deltas.get(g, [0])) for g in groups_ordered]
    cols_g = [GROUP_COLORS.get(g, '#95a5a6') for g in groups_ordered]
    ax2.barh(y_g2, vals_g, color=cols_g, alpha=0.85, edgecolor='white')
    ax2.set_yticks(y_g2)
    ax2.set_yticklabels([g.replace(' ', '\n') for g in groups_ordered], fontsize=9)
    ax2.set_xlabel('Mean Δ (λ=0.01)')
    ax2.set_title('WCA Advantage by Concept Type', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.invert_yaxis()
    
    # Panel 3: Per-pair bars
    ax3 = fig.add_subplot(gs[0, 2])
    pair_means = {}
    for pair in ['en-fr', 'es-de', 'fr-de', 'fr-es']:
        deltas_p = [r['mean_delta'] for r in results_by_lambda[0.01] if r['pair'] == pair]
        pair_means[pair] = np.mean(deltas_p) if deltas_p else 0
    pairs_sorted = sorted(pair_means.keys(), key=lambda p: pair_means[p], reverse=True)
    vals_p = [pair_means[p] for p in pairs_sorted]
    c_p = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals_p]
    ax3.barh(range(len(pairs_sorted)), vals_p, color=c_p, alpha=0.85, edgecolor='white')
    ax3.set_yticks(range(len(pairs_sorted)))
    ax3.set_yticklabels([LANG_PAIR_LABELS.get(p, p) for p in pairs_sorted], fontsize=11)
    ax3.set_xlabel('Mean Δ (λ=0.01)')
    ax3.set_title('WCA Advantage by Language Pair', fontsize=12, fontweight='bold')
    ax3.axvline(0, color='black', linewidth=1)
    ax3.invert_yaxis()
    
    # Panel 4: Per-model heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    model_names = sorted(set(r['model'] for r in results_by_lambda[0.01]))
    pair_names = ['en-fr', 'es-de', 'fr-de', 'fr-es']
    grid = np.full((len(model_names), len(pair_names)), np.nan)
    for r in results_by_lambda[0.01]:
        i = model_names.index(r['model'])
        j = pair_names.index(r['pair'])
        grid[i, j] = r['mean_delta']
    
    sns.heatmap(grid, annot=True, fmt="+.3f", cmap="PiYG", center=0,
                xticklabels=[LANG_PAIR_LABELS.get(p, p) for p in pair_names],
                yticklabels=model_names, ax=ax4,
                linewidths=0.5, linecolor='white', vmin=-0.05, vmax=0.05,
                annot_kws={"size": 9})
    ax4.set_title('Δ(WCA − Naive) per Model × Pair (λ=0.01)', fontsize=12, fontweight='bold')
    
    # Panel 5: Summary stats
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    wins_01 = sum(1 for r in results_by_lambda[0.01] if r['mean_delta'] > 0)
    total_01 = len(results_by_lambda[0.01])
    mean_01, lo_01, hi_01 = bootstrap_ci([r['mean_delta'] for r in results_by_lambda[0.01]])
    
    table_data = [
        ['Models', str(len(model_names))],
        ['Language Pairs', '4'],
        ['Total Comparisons', str(total_01)],
        ['', ''],
        ['λ=0.001 Win Rate', f"{sum(1 for r in results_by_lambda[0.001] if r['mean_delta'] > 0)}/{len(results_by_lambda[0.001])}"],
        ['λ=0.01 Win Rate', f"{wins_01}/{total_01} ({100*wins_01/total_01:.0f}%)"],
        ['λ=0.1 Win Rate', f"{sum(1 for r in results_by_lambda[0.1] if r['mean_delta'] > 0)}/{len(results_by_lambda[0.1])}"],
        ['', ''],
        ['Mean Δ (λ=0.01)', f'{mean_01:+.4f}'],
        ['95% CI', f'[{lo_01:+.4f}, {hi_01:+.4f}]'],
    ]
    if aniso_data:
        table_data.append(['Aniso. Correlation', f'r={r_val:.3f}, p={p_val:.2e}'])
    
    table = ax5.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left', colWidths=[0.55, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif table_data[row-1][0] == '':
            cell.set_facecolor('white')
            cell.set_edgecolor('white')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')
    ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle('RAID Experiment — Deep Analysis Dashboard', fontsize=18, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(output_dir, "deep_analysis_dashboard.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: deep_analysis_dashboard.png")
    
    print(f"\n{'='*70}")
    print(f"  Analysis complete! All outputs in: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    results_root = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(results_root)
