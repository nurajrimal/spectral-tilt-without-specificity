#!/usr/bin/env python3
"""
Hierarchy Experiments — Direct tests of the causal geometry hypothesis.

All experiments run from saved .pt files (no GPU needed).
  1. Concept Orthogonality: Euclidean vs Causal space
  2. Entanglement Reduction Score
  3. Spectral Alignment of Concepts
  4. Cross-Lingual Angle Preservation
  5. Whitening Sensitivity (per-eigenvalue analysis)

Usage: python scripts/hierarchy_experiments.py results/
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
import io
import glob
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import linear_rep_geometry as lrg

sns.set_theme(context="paper", style="white", palette="colorblind", font="sans-serif", font_scale=1.4)

# ──────────────────────────────────────────────────────────
# Quiet whitening helper (suppresses condition number spam)
# ──────────────────────────────────────────────────────────
def quiet_whitening(cov, reg_lambda=0.1, truncate_k=None):
    """Wrapper around lrg.get_whitening_transform that suppresses stdout."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = lrg.get_whitening_transform(cov, reg_lambda=reg_lambda, truncate_k=truncate_k)
    finally:
        sys.stdout = old_stdout
    return result

# ──────────────────────────────────────────────────────────
# Concept Groups (for cross-group vs within-group analysis)
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
    'Other':           '#95a5a6',
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

def load_vectors(pair_dir):
    """Load concept vectors from a pair directory."""
    vectors = {}
    for f in glob.glob(os.path.join(pair_dir, "v_raw_*.pt")):
        name = os.path.basename(f).replace("v_raw_", "").replace(".pt", "")
        vectors[name] = torch.load(f, map_location='cpu', weights_only=True).float()
    return vectors

def load_transported(pair_dir):
    """Load WCA and naive transported vectors."""
    raid, naive = {}, {}
    for f in glob.glob(os.path.join(pair_dir, "v_raid_*.pt")):
        name = os.path.basename(f).replace("v_raid_", "").replace(".pt", "")
        raid[name] = torch.load(f, map_location='cpu', weights_only=True).float()
    for f in glob.glob(os.path.join(pair_dir, "v_naive_*.pt")):
        name = os.path.basename(f).replace("v_naive_", "").replace(".pt", "")
        naive[name] = torch.load(f, map_location='cpu', weights_only=True).float()
    return raid, naive

def load_covariance(pair_dir):
    """Load source covariance."""
    for name in ["cov_src.pt", "cov_en.pt"]:
        p = os.path.join(pair_dir, name)
        if os.path.exists(p):
            return torch.load(p, map_location='cpu', weights_only=True).float()
    return None

def load_activations(pair_dir):
    """Load source and target activations."""
    X_src = X_tgt = None
    for sn, tn in [("X_src.pt", "X_tgt.pt"), ("X_en.pt", "X_fr.pt")]:
        sp, tp = os.path.join(pair_dir, sn), os.path.join(pair_dir, tn)
        if os.path.exists(sp) and os.path.exists(tp):
            X_src = torch.load(sp, map_location='cpu', weights_only=True).float()
            X_tgt = torch.load(tp, map_location='cpu', weights_only=True).float()
            break
    cov_tgt = None
    for name in ["cov_tgt.pt", "cov_fr.pt"]:
        p = os.path.join(pair_dir, name)
        if os.path.exists(p):
            cov_tgt = torch.load(p, map_location='cpu', weights_only=True).float()
            break
    return X_src, X_tgt, cov_tgt


# ══════════════════════════════════════════════════════════
# EXPERIMENT 1: Concept Orthogonality
# ══════════════════════════════════════════════════════════
def experiment_1_orthogonality(all_vectors, all_covs, output_dir):
    print("\n" + "="*70)
    print("  EXPERIMENT 1: Concept Orthogonality — Euclidean vs Causal")
    print("="*70)
    
    all_euc_cross, all_cau_cross = [], []
    all_euc_within, all_cau_within = [], []
    per_model = []
    
    keys = sorted(all_vectors.keys())
    for ki, key in enumerate(keys):
        vectors = all_vectors[key]
        cov = all_covs.get(key)
        if cov is None or len(vectors) < 3:
            continue
        
        print(f"    [{ki+1}/{len(keys)}] {key[0]}/{key[1]}...", end='\r')
        # Compute whitening transform
        try:
            psi, _ = quiet_whitening(cov, reg_lambda=0.1)
        except Exception:
            continue
        
        names = sorted(vectors.keys())
        euc_cross, cau_cross = [], []
        euc_within, cau_within = [], []
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                v_i, v_j = vectors[names[i]], vectors[names[j]]
                g_i, g_j = get_group(names[i]), get_group(names[j])
                
                # Euclidean cosine
                euc = abs(cosine_sim(v_i, v_j))
                
                # Causal cosine (whitened)
                w_i = psi @ v_i
                w_j = psi @ v_j
                cau = abs(cosine_sim(w_i, w_j))
                
                if g_i == g_j and g_i != 'Other':
                    euc_within.append(euc)
                    cau_within.append(cau)
                    all_euc_within.append(euc)
                    all_cau_within.append(cau)
                else:
                    euc_cross.append(euc)
                    cau_cross.append(cau)
                    all_euc_cross.append(euc)
                    all_cau_cross.append(cau)
        
        model, pair = key
        me_cross = np.mean(euc_cross) if euc_cross else 0
        mc_cross = np.mean(cau_cross) if cau_cross else 0
        me_within = np.mean(euc_within) if euc_within else 0
        mc_within = np.mean(cau_within) if cau_within else 0
        
        per_model.append({
            'key': key, 'model': model, 'pair': pair,
            'euc_cross': me_cross, 'cau_cross': mc_cross,
            'euc_within': me_within, 'cau_within': mc_within,
            'reduction_cross': (me_cross - mc_cross) / max(me_cross, 1e-10),
            'reduction_within': (me_within - mc_within) / max(me_within, 1e-10),
        })
    
    # Summary
    me_c = np.mean(all_euc_cross)
    mc_c = np.mean(all_cau_cross)
    me_w = np.mean(all_euc_within)
    mc_w = np.mean(all_cau_within)
    
    print(f"\n  Cross-group |cos|:")
    print(f"    Euclidean: {me_c:.4f}")
    print(f"    Causal:    {mc_c:.4f}  {'↓ REDUCED' if mc_c < me_c else '↑ INCREASED'} ({100*(me_c-mc_c)/me_c:+.1f}%)")
    print(f"\n  Within-group |cos|:")
    print(f"    Euclidean: {me_w:.4f}")
    print(f"    Causal:    {mc_w:.4f}  {'↓ REDUCED' if mc_w < me_w else '↑ INCREASED'} ({100*(me_w-mc_w)/me_w:+.1f}%)")
    
    # Statistical test
    t_cross, p_cross = sp_stats.ttest_rel(all_euc_cross, all_cau_cross)
    t_within, p_within = sp_stats.ttest_rel(all_euc_within, all_cau_within)
    print(f"\n  Paired t-test (cross-group):  t={t_cross:.3f}, p={p_cross:.2e} {'✓' if p_cross < 0.05 else '✗'}")
    print(f"  Paired t-test (within-group): t={t_within:.3f}, p={p_within:.2e} {'✓' if p_within < 0.05 else '✗'}")
    
    # Key test: does whitening disentangle cross-group MORE than within-group?
    reductions_cross = [d['reduction_cross'] for d in per_model]
    reductions_within = [d['reduction_within'] for d in per_model]
    t_diff, p_diff = sp_stats.ttest_ind(reductions_cross, reductions_within)
    print(f"\n  Selectivity test (cross vs within reduction): t={t_diff:.3f}, p={p_diff:.2e} {'✓' if p_diff < 0.05 else '✗'}")
    print(f"  Mean reduction — Cross: {np.mean(reductions_cross)*100:.1f}%  Within: {np.mean(reductions_within)*100:.1f}%")
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Scatter — Euclidean vs Causal (cross-group)
    ax = axes[0]
    ax.scatter(all_euc_cross, all_cau_cross, alpha=0.05, s=3, c='#e74c3c', rasterized=True)
    lim = max(max(all_euc_cross), max(all_cau_cross)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('|cos| Euclidean')
    ax.set_ylabel('|cos| Causal (Σ⁻¹)')
    ax.set_title('Cross-Group Concept Pairs', fontweight='bold')
    ax.legend()
    # Count above/below diagonal
    above = sum(1 for e, c in zip(all_euc_cross, all_cau_cross) if c > e)
    below = len(all_euc_cross) - above
    ax.text(0.05, 0.95, f'Reduced: {below}\nIncreased: {above}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Panel 2: Scatter — within-group
    ax = axes[1]
    ax.scatter(all_euc_within, all_cau_within, alpha=0.05, s=3, c='#3498db', rasterized=True)
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('|cos| Euclidean')
    ax.set_ylabel('|cos| Causal (Σ⁻¹)')
    ax.set_title('Within-Group Concept Pairs', fontweight='bold')
    ax.legend()
    above_w = sum(1 for e, c in zip(all_euc_within, all_cau_within) if c > e)
    below_w = len(all_euc_within) - above_w
    ax.text(0.05, 0.95, f'Reduced: {below_w}\nIncreased: {above_w}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Panel 3: Per-model reduction comparison
    ax = axes[2]
    models_sorted = sorted(per_model, key=lambda d: d['reduction_cross'], reverse=True)
    model_labels = [f"{d['model'][:12]}" for d in models_sorted[:20]]
    r_cross = [d['reduction_cross']*100 for d in models_sorted[:20]]
    r_within = [d['reduction_within']*100 for d in models_sorted[:20]]
    y = np.arange(len(model_labels))
    ax.barh(y - 0.15, r_cross, 0.3, label='Cross-group', color='#e74c3c', alpha=0.8)
    ax.barh(y + 0.15, r_within, 0.3, label='Within-group', color='#3498db', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_xlabel('Entanglement Reduction (%)')
    ax.set_title('Reduction by Model', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    
    plt.suptitle('Exp 1: Does Whitening Disentangle Concepts?', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_orthogonality.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: exp1_orthogonality.png")
    
    return {
        'euc_cross': me_c, 'cau_cross': mc_c,
        'euc_within': me_w, 'cau_within': mc_w,
        'p_cross': p_cross, 'p_within': p_within,
        'p_selectivity': p_diff,
        'n_cross': len(all_euc_cross), 'n_within': len(all_euc_within),
    }


# ══════════════════════════════════════════════════════════
# EXPERIMENT 2: Entanglement Reduction Score
# ══════════════════════════════════════════════════════════
def experiment_2_entanglement(all_vectors, all_covs, output_dir):
    print("\n" + "="*70)
    print("  EXPERIMENT 2: Entanglement Reduction Score")
    print("="*70)
    
    results = []
    lambdas = [0.001, 0.01, 0.1, 1.0]
    
    keys = sorted(all_vectors.keys())
    for ki, key in enumerate(keys):
        vectors = all_vectors[key]
        cov = all_covs.get(key)
        if cov is None or len(vectors) < 3:
            continue
        
        print(f"    [{ki+1}/{len(keys)}] {key[0]}/{key[1]}...", end='\r')
        names = sorted(vectors.keys())
        
        # Euclidean entanglement
        euc_sims = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                euc_sims.append(abs(cosine_sim(vectors[names[i]], vectors[names[j]])))
        E_euc = np.mean(euc_sims)
        
        model, pair = key
        row = {'model': model, 'pair': pair, 'E_euc': E_euc}
        
        for lam in lambdas:
            try:
                psi, _ = quiet_whitening(cov, reg_lambda=lam)
                whitened = {n: psi @ v for n, v in vectors.items()}
                cau_sims = []
                for i in range(len(names)):
                    for j in range(i+1, len(names)):
                        cau_sims.append(abs(cosine_sim(whitened[names[i]], whitened[names[j]])))
                E_cau = np.mean(cau_sims)
                row[f'E_cau_{lam}'] = E_cau
                row[f'reduction_{lam}'] = (E_euc - E_cau) / max(E_euc, 1e-10)
            except Exception:
                row[f'E_cau_{lam}'] = E_euc
                row[f'reduction_{lam}'] = 0
        
        results.append(row)
    
    # Summary
    print(f"\n  {'λ':<8} {'Mean Entanglement':>18} {'Mean Reduction':>15}")
    print(f"  {'-'*45}")
    print(f"  {'Euclid':<8} {np.mean([r['E_euc'] for r in results]):>18.4f} {'(baseline)':>15}")
    for lam in lambdas:
        me = np.mean([r.get(f'E_cau_{lam}', 0) for r in results])
        mr = np.mean([r.get(f'reduction_{lam}', 0) for r in results])
        sig = sp_stats.ttest_1samp([r.get(f'reduction_{lam}', 0) for r in results], 0)
        print(f"  {lam:<8} {me:>18.4f} {mr*100:>+14.1f}% p={sig.pvalue:.2e} {'✓' if sig.pvalue < 0.05 else '✗'}")
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Bar chart of entanglement by λ
    ax = axes[0]
    x = np.arange(len(lambdas) + 1)
    vals = [np.mean([r['E_euc'] for r in results])]
    vals += [np.mean([r.get(f'E_cau_{lam}', 0) for r in results]) for lam in lambdas]
    labels = ['Euclidean'] + [f'λ={l}' for l in lambdas]
    colors = ['#95a5a6'] + ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Mean |cos| Between Concepts', fontsize=12)
    ax.set_title('Concept Entanglement by Metric', fontweight='bold')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
    
    # Panel 2: Per-model reduction at λ=0.1
    ax = axes[1]
    models_unique = sorted(set(r['model'] for r in results))
    model_means = []
    for m in models_unique:
        reds = [r['reduction_0.1'] for r in results if r['model'] == m]
        model_means.append((m, np.mean(reds)))
    model_means.sort(key=lambda x: x[1], reverse=True)
    
    y = np.arange(len(model_means))
    vals_m = [v for _, v in model_means]
    colors_m = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals_m]
    ax.barh(y, [v*100 for v in vals_m], color=colors_m, alpha=0.85, edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in model_means], fontsize=9)
    ax.set_xlabel('Entanglement Reduction (%)', fontsize=12)
    ax.set_title('Per-Model Reduction (λ=0.1)', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    
    plt.suptitle('Exp 2: Does Whitening Reduce Concept Entanglement?', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_entanglement.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: exp2_entanglement.png")
    
    return results


# ══════════════════════════════════════════════════════════
# EXPERIMENT 3: Spectral Alignment
# ══════════════════════════════════════════════════════════
def experiment_3_spectral(all_vectors, all_covs, output_dir):
    print("\n" + "="*70)
    print("  EXPERIMENT 3: Spectral Alignment of Concepts")
    print("="*70)
    
    # For each model, check where concept vectors live in eigenspace of Σ
    all_energy_profiles = []  # (fraction of energy in top-k% eigenvalues)
    
    for key in sorted(all_vectors.keys()):
        vectors = all_vectors[key]
        cov = all_covs.get(key)
        if cov is None or len(vectors) < 3:
            continue
        
        # Eigendecompose
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        d = len(eigenvalues)
        model, pair = key
        
        for name, v in vectors.items():
            v = v.flatten()
            # Project onto eigenvectors
            projections = eigenvectors.T @ v  # (d,)
            energies = projections ** 2
            total_energy = energies.sum().item()
            
            if total_energy < 1e-10:
                continue
            
            # Cumulative energy in top-k%
            cumulative = torch.cumsum(energies, dim=0) / total_energy
            
            # Energy in top 10%, 25%, 50%
            top10 = cumulative[max(0, int(d*0.1)-1)].item()
            top25 = cumulative[max(0, int(d*0.25)-1)].item()
            top50 = cumulative[max(0, int(d*0.5)-1)].item()
            
            all_energy_profiles.append({
                'model': model, 'pair': pair, 'concept': name,
                'group': get_group(name),
                'top10': top10, 'top25': top25, 'top50': top50,
            })
    
    # Summary
    mean_top10 = np.mean([p['top10'] for p in all_energy_profiles])
    mean_top25 = np.mean([p['top25'] for p in all_energy_profiles])
    mean_top50 = np.mean([p['top50'] for p in all_energy_profiles])
    
    print(f"\n  Concept energy in Σ eigenspace:")
    print(f"    Top 10% eigenvalues contain: {mean_top10*100:.1f}% of concept energy")
    print(f"    Top 25% eigenvalues contain: {mean_top25*100:.1f}% of concept energy")
    print(f"    Top 50% eigenvalues contain: {mean_top50*100:.1f}% of concept energy")
    
    # If uniform: top-k% should contain k% of energy
    print(f"\n  Comparison to uniform distribution:")
    print(f"    Top 10%: {mean_top10*100:.1f}% vs 10% expected → {mean_top10*100/10:.1f}x concentration")
    print(f"    Top 25%: {mean_top25*100:.1f}% vs 25% expected → {mean_top25*100/25:.1f}x concentration")
    print(f"    Top 50%: {mean_top50*100:.1f}% vs 50% expected → {mean_top50*100/50:.1f}x concentration")
    
    high_conc = mean_top10 > 0.3  # More than 30% in top 10%
    print(f"\n  {'→ Concepts CONCENTRATED in high-eigenvalue directions (noise)' if high_conc else '→ Concepts SPREAD across eigenvalues (supports disentanglement theory)'}")
    
    # Per-group analysis
    print(f"\n  Per-group energy in top-10% eigenvalues:")
    for g in ['Verb Morphology', 'Adj Morphology', 'Noun Morphology', 'Language Pairs', 'Semantic']:
        vals = [p['top10'] for p in all_energy_profiles if p['group'] == g]
        if vals:
            print(f"    {g:<20}: {np.mean(vals)*100:.1f}%")
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Distribution of top-10% energy
    ax = axes[0]
    vals = [p['top10']*100 for p in all_energy_profiles]
    ax.hist(vals, bins=40, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(10, color='red', linestyle='--', linewidth=2, label='Uniform (10%)')
    ax.axvline(np.mean(vals), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(vals):.1f}%)')
    ax.set_xlabel('% of Energy in Top-10% Eigenvalues')
    ax.set_ylabel('Count')
    ax.set_title('Energy Concentration', fontweight='bold')
    ax.legend()
    
    # Panel 2: Group comparison
    ax = axes[1]
    groups = ['Verb Morphology', 'Adj Morphology', 'Noun Morphology', 'Language Pairs', 'Semantic']
    group_vals = []
    for g in groups:
        gv = [p['top10']*100 for p in all_energy_profiles if p['group'] == g]
        group_vals.append(gv)
    
    bp = ax.boxplot(group_vals, tick_labels=[g.replace(' ', '\n') for g in groups],
                     patch_artist=True, showmeans=True)
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(GROUP_COLORS.get(g, '#95a5a6'))
        patch.set_alpha(0.7)
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax.set_ylabel('% Energy in Top-10% Eigenvalues')
    ax.set_title('By Concept Group', fontweight='bold')
    ax.legend()
    
    # Panel 3: Cumulative energy curve (average across all)
    ax = axes[2]
    # Pick a representative model
    rep_key = list(all_vectors.keys())[0]
    cov_rep = all_covs[rep_key]
    evals_rep, evecs_rep = torch.linalg.eigh(cov_rep)
    idx = torch.argsort(evals_rep, descending=True)
    evals_rep = evals_rep[idx]
    evecs_rep = evecs_rep[:, idx]
    d = len(evals_rep)
    
    # Average cumulative energy across concepts
    cumul_all = []
    for name, v in all_vectors[rep_key].items():
        v = v.flatten()
        proj = evecs_rep.T @ v
        e = (proj ** 2)
        e = e / e.sum().clamp(min=1e-10)
        cumul_all.append(torch.cumsum(e, dim=0).detach().numpy())
    
    mean_cumul = np.mean(cumul_all, axis=0)
    x_frac = np.arange(1, d+1) / d * 100
    ax.plot(x_frac, mean_cumul * 100, 'b-', linewidth=2, label='Concept vectors')
    ax.plot(x_frac, x_frac, 'r--', alpha=0.5, linewidth=1.5, label='Uniform')
    ax.set_xlabel('Top-k% Eigenvalues')
    ax.set_ylabel('Cumulative Energy (%)')
    ax.set_title(f'Energy Distribution ({rep_key[0]})', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    
    plt.suptitle('Exp 3: Where Do Concepts Live in the Eigenspectrum?', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp3_spectral.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: exp3_spectral.png")
    
    return all_energy_profiles


# ══════════════════════════════════════════════════════════
# EXPERIMENT 4: Cross-Lingual Angle Preservation
# ══════════════════════════════════════════════════════════
def experiment_4_angle_preservation(all_vectors, all_raid, all_naive, output_dir):
    print("\n" + "="*70)
    print("  EXPERIMENT 4: Cross-Lingual Angle Preservation")
    print("="*70)
    
    wca_angle_errors = []
    naive_angle_errors = []
    
    for key in sorted(all_vectors.keys()):
        vectors = all_vectors[key]
        raid = all_raid.get(key, {})
        naive = all_naive.get(key, {})
        
        common = sorted(set(vectors.keys()) & set(raid.keys()) & set(naive.keys()))
        if len(common) < 3:
            continue
        
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                a, b = common[i], common[j]
                
                # Source angle
                theta_src = cosine_sim(vectors[a], vectors[b])
                
                # WCA angle
                theta_wca = cosine_sim(raid[a], raid[b])
                
                # Naive angle
                theta_naive = cosine_sim(naive[a], naive[b])
                
                wca_angle_errors.append(abs(theta_src - theta_wca))
                naive_angle_errors.append(abs(theta_src - theta_naive))
    
    mean_wca = np.mean(wca_angle_errors)
    mean_naive = np.mean(naive_angle_errors)
    t, p = sp_stats.ttest_rel(wca_angle_errors, naive_angle_errors)
    
    print(f"\n  Mean angle error:")
    print(f"    WCA:   {mean_wca:.4f}")
    print(f"    Naive: {mean_naive:.4f}")
    print(f"    Δ = {mean_wca - mean_naive:+.4f} ({'WCA better' if mean_wca < mean_naive else 'Naive better'})")
    print(f"    Paired t-test: t={t:.3f}, p={p:.2e} {'✓' if p < 0.05 else '✗'}")
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Histogram comparison
    ax = axes[0]
    bins = np.linspace(0, max(max(wca_angle_errors), max(naive_angle_errors)), 50)
    ax.hist(naive_angle_errors, bins=bins, alpha=0.5, color='#e74c3c', label=f'Naive (μ={mean_naive:.4f})', density=True)
    ax.hist(wca_angle_errors, bins=bins, alpha=0.5, color='#2ecc71', label=f'WCA (μ={mean_wca:.4f})', density=True)
    ax.set_xlabel('|θ_source − θ_target|')
    ax.set_ylabel('Density')
    ax.set_title('Angle Error Distribution', fontweight='bold')
    ax.legend()
    
    # Panel 2: Scatter
    ax = axes[1]
    sample_idx = np.random.choice(len(wca_angle_errors), min(5000, len(wca_angle_errors)), replace=False)
    ax.scatter([naive_angle_errors[i] for i in sample_idx],
               [wca_angle_errors[i] for i in sample_idx],
               alpha=0.05, s=3, c='#3498db', rasterized=True)
    lim = max(max(naive_angle_errors), max(wca_angle_errors)) * 0.8
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Naive Angle Error')
    ax.set_ylabel('WCA Angle Error')
    ax.set_title('WCA vs Naive (per concept pair)', fontweight='bold')
    below = sum(1 for w, n in zip(wca_angle_errors, naive_angle_errors) if w < n)
    ax.text(0.05, 0.95, f'WCA better: {below}/{len(wca_angle_errors)}\n({100*below/len(wca_angle_errors):.0f}%)',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.legend()
    
    plt.suptitle('Exp 4: Which Method Preserves Inter-Concept Angles Better?', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp4_angle_preservation.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: exp4_angle_preservation.png")
    
    return {'wca_error': mean_wca, 'naive_error': mean_naive, 'p': p}


# ══════════════════════════════════════════════════════════
# EXPERIMENT 5: Whitening Sensitivity
# ══════════════════════════════════════════════════════════
def experiment_5_sensitivity(all_vectors, all_covs, all_activations, output_dir):
    print("\n" + "="*70)
    print("  EXPERIMENT 5: Whitening Sensitivity — Progressive Truncation")
    print("="*70)
    
    truncation_fracs = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    results = defaultdict(list)
    
    for key in sorted(all_vectors.keys()):
        vectors = all_vectors[key]
        cov_src = all_covs.get(key)
        act_data = all_activations.get(key)
        
        if cov_src is None or act_data is None or len(vectors) < 3:
            continue
        
        X_src, X_tgt, cov_tgt = act_data
        if X_src is None or cov_tgt is None:
            continue
        
        model, pair = key
        d = cov_src.shape[0]
        
        # Compute naive baseline
        naive_sims = []
        # Solve naive procrustes
        try:
            U, S, Vt = torch.linalg.svd(X_src.T @ X_tgt)
            Q_naive = U @ Vt
        except Exception:
            continue
        
        for name, v in vectors.items():
            v_naive = Q_naive @ v
            naive_sims.append(cosine_sim(v, v_naive))
        mean_naive = np.mean(naive_sims)
        
        # WCA with progressive truncation
        for frac in truncation_fracs:
            k = max(1, int(d * frac))
            try:
                psi_src, _ = quiet_whitening(cov_src, reg_lambda=0.1, truncate_k=k)
                psi_tgt, sqrt_tgt = quiet_whitening(cov_tgt, reg_lambda=0.1, truncate_k=k)
                Q = lrg.solve_whitened_procrustes(X_src, X_tgt, psi_src, psi_tgt)
                
                wca_sims = []
                for name, v in vectors.items():
                    v_wca = lrg.transport_concept_vector(v, psi_src, sqrt_tgt, Q)
                    wca_sims.append(cosine_sim(v, v_wca))
                
                mean_wca = np.mean(wca_sims)
                delta = mean_wca - mean_naive
                results[frac].append({'model': model, 'pair': pair, 'delta': delta, 'wca': mean_wca, 'naive': mean_naive})
            except Exception as e:
                pass
    
    # Summary
    print(f"\n  {'Truncation':>12} {'Mean Δ':>10} {'WCA Wins':>10}")
    print(f"  {'-'*35}")
    for frac in truncation_fracs:
        if results[frac]:
            md = np.mean([r['delta'] for r in results[frac]])
            wins = sum(1 for r in results[frac] if r['delta'] > 0)
            total = len(results[frac])
            print(f"  {frac*100:>10.0f}% {md:>+10.4f} {wins:>4}/{total} ({100*wins/total:.0f}%)")
    
    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fracs_pct = [f*100 for f in truncation_fracs]
    means = [np.mean([r['delta'] for r in results[f]]) if results[f] else 0 for f in truncation_fracs]
    wins_pct = [sum(1 for r in results[f] if r['delta'] > 0) / max(len(results[f]), 1) * 100 
                if results[f] else 50 for f in truncation_fracs]
    
    ax.plot(fracs_pct, means, 'bo-', linewidth=2, markersize=8, label='Mean Δ(WCA−Naive)')
    ax.fill_between(fracs_pct, means, alpha=0.1, color='blue')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Eigenvalue Truncation (% of dimensions kept)')
    ax.set_ylabel('Δ(WCA − Naive)')
    ax.set_title('Exp 5: Effect of Eigenvalue Truncation on WCA Advantage', fontsize=14, fontweight='bold')
    
    ax2 = ax.twinx()
    ax2.plot(fracs_pct, wins_pct, 'r^--', linewidth=1.5, markersize=6, alpha=0.7, label='Win Rate %')
    ax2.axhline(50, color='red', linewidth=0.5, alpha=0.3)
    ax2.set_ylabel('WCA Win Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp5_sensitivity.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: exp5_sensitivity.png")
    
    return results


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main(results_root):
    output_dir = os.path.join(results_root, "_hierarchy")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"  HIERARCHY EXPERIMENTS")
    print(f"  Results: {results_root}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*70}")
    
    # ── Load all data ──
    all_vectors = {}
    all_covs = {}
    all_raid = {}
    all_naive = {}
    all_activations = {}
    
    models = sorted([d for d in os.listdir(results_root)
                     if os.path.isdir(os.path.join(results_root, d)) and not d.startswith('_')])
    
    for model in models:
        mdir = os.path.join(results_root, model)
        for pair in ['en-fr', 'es-de', 'fr-de', 'fr-es']:
            pdir = os.path.join(mdir, pair)
            if not os.path.exists(os.path.join(pdir, "stats.txt")):
                continue
            
            key = (model, pair)
            print(f"  Loading {model}/{pair}...")
            
            all_vectors[key] = load_vectors(pdir)
            all_covs[key] = load_covariance(pdir)
            raid, naive = load_transported(pdir)
            all_raid[key] = raid
            all_naive[key] = naive
            X_src, X_tgt, cov_tgt = load_activations(pdir)
            all_activations[key] = (X_src, X_tgt, cov_tgt)
    
    print(f"\n  Loaded {len(all_vectors)} model×pair combinations\n")
    
    # Run experiments
    r1 = experiment_1_orthogonality(all_vectors, all_covs, output_dir)
    r2 = experiment_2_entanglement(all_vectors, all_covs, output_dir)
    r3 = experiment_3_spectral(all_vectors, all_covs, output_dir)
    r4 = experiment_4_angle_preservation(all_vectors, all_raid, all_naive, output_dir)
    r5 = experiment_5_sensitivity(all_vectors, all_covs, all_activations, output_dir)
    
    # ══════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT: Is There Evidence of Hierarchy?")
    print(f"{'='*70}")
    print(f"\n  Exp 1 — Orthogonality:")
    print(f"    Cross-group: Euc={r1['euc_cross']:.4f} → Causal={r1['cau_cross']:.4f} (p={r1['p_cross']:.2e})")
    print(f"    {'✓ Whitening DISENTANGLES cross-group concepts' if r1['p_cross'] < 0.05 and r1['cau_cross'] < r1['euc_cross'] else '✗ No significant disentanglement'}")
    
    print(f"\n  Exp 4 — Angle Preservation:")
    print(f"    WCA error={r4['wca_error']:.4f}, Naive error={r4['naive_error']:.4f} (p={r4['p']:.2e})")
    print(f"    {'✓ WCA preserves structure better' if r4['p'] < 0.05 and r4['wca_error'] < r4['naive_error'] else '✗ No significant difference' if r4['p'] >= 0.05 else '✗ Naive preserves structure better'}")
    
    print(f"\n{'='*70}")
    print(f"  All outputs in: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    results_root = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(results_root)
