#!/usr/bin/env python3
"""
V2 experiment suite for spectral anti-concentration robustness, nulls,
layer-wise trajectories, and behavioral steering examples.
"""

import argparse
import csv
import gc
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import linear_rep_geometry as lrg

try:
    from fix_hole4_v2_split_injection import (
        CONCEPT_TESTS,
        get_layers,
        decompose_vector,
        load_heldout_passages,
        compute_perplexity_heldout,
        compute_logit_shift,
        MultiLayerSteeringContext,
    )
except Exception:
    CONCEPT_TESTS = {}
    get_layers = None
    decompose_vector = None
    load_heldout_passages = None
    compute_perplexity_heldout = None
    compute_logit_shift = None
    MultiLayerSteeringContext = None

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/_spectral/v2_experiments")
EIGEN_CACHE_DIR = Path("results/_spectral/eigen_cache")

DEFAULT_MODEL_PATHS = {
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "jetmoe-8b": "jetmoe/jetmoe-8b",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
    "OLMoE-1B-7B-0125": "allenai/OLMoE-1B-7B-0125",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
}

ALL_EXPERIMENTS = [
    "covariance",
    "lambda",
    "nulls",
    "layerwise",
    "behavioral",
    "sae_controls",
]

CONCEPT_POS_CLASS = {
    "verb": ["[verb - Ved]", "[verb - Ving]", "[verb - 3pSg]", "[verb - V + er]", "[verb - V + able]", "[verb - V + ment]", "[verb - V + tion]", "[Ving - Ved]", "[Ving - 3pSg]", "[3pSg - Ved]"],
    "adj": ["[adj - comparative]", "[adj - superlative]", "[adj - un + adj]", "[adj - adj + ly]"],
    "noun": ["[noun - plural]", "[pronoun - possessive]"],
    "language": ["[English - French]", "[French - German]", "[French - Spanish]", "[German - Spanish]"],
    "semantic": ["[male - female]", "[country - capital]", "[thing - color]", "[thing - part]", "[small - big]", "[lower - upper]", "[frequent - infrequent]"],
}


def safe_name(name):
    return name.replace("/", "__").replace("[", "").replace("]", "").replace(" ", "_")


def model_name_from_path(model):
    return model.split("/")[-1]


def model_path_from_arg(model):
    return model if "/" in model else DEFAULT_MODEL_PATHS.get(model, model)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_csv(path, rows, fieldnames=None):
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_dir_for(model_name, pair):
    pair_dir = RESULTS_DIR / model_name / pair
    if pair_dir.exists():
        return pair_dir
    return RESULTS_DIR / model_name


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True).float()


def load_covariance(model_name, pair="en-fr", side="src"):
    pair_dir = pair_dir_for(model_name, pair)
    names = [f"cov_{side}.pt"]
    if side == "src":
        names.append("cov_en.pt")
    elif side == "tgt":
        names.append("cov_fr.pt")
    for name in names:
        path = pair_dir / name
        if path.exists():
            return load_tensor(path), path
    raise FileNotFoundError(f"No {side} covariance for {model_name}/{pair}")


def load_alignment_data(model_name, pair="en-fr"):
    pair_dir = pair_dir_for(model_name, pair)
    cov_src, _ = load_covariance(model_name, pair, "src")
    cov_tgt, _ = load_covariance(model_name, pair, "tgt")
    x_src = x_tgt = None
    for sn, tn in [("X_src.pt", "X_tgt.pt"), ("X_en.pt", "X_fr.pt")]:
        if (pair_dir / sn).exists() and (pair_dir / tn).exists():
            x_src = load_tensor(pair_dir / sn)
            x_tgt = load_tensor(pair_dir / tn)
            break
    if x_src is None:
        raise FileNotFoundError(f"No X_src/X_tgt activations for {model_name}/{pair}")
    vectors = load_concept_vectors(model_name, pair)
    if not vectors:
        raise FileNotFoundError(f"No v_raw concept vectors for {model_name}/{pair}")
    return pair_dir, cov_src, cov_tgt, x_src, x_tgt, vectors


def load_concept_vectors(model_name, pair="en-fr"):
    pair_dir = pair_dir_for(model_name, pair)
    vectors = {}
    for path in sorted(pair_dir.glob("v_raw_*.pt")):
        vectors[path.stem.replace("v_raw_", "")] = load_tensor(path)
    return vectors


def load_word_pairs(concept_name):
    path = Path("data/word_pairs") / f"{concept_name}.txt"
    pairs = []
    if not path.exists():
        return pairs
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def all_word_pair_concepts():
    return [Path(p).stem for p in sorted(glob.glob("data/word_pairs/*.txt"))]


def concept_class(concept_name):
    for cls, concepts in CONCEPT_POS_CLASS.items():
        if concept_name in concepts:
            return cls
    return "other"


def eig_sorted(cov):
    cov = cov.float()
    try:
        vals, vecs = torch.linalg.eigh(cov)
    except RuntimeError:
        if not torch.cuda.is_available():
            raise
        cov_gpu = cov.cuda()
        vals, vecs = torch.linalg.eigh(cov_gpu)
        vals, vecs = vals.cpu(), vecs.cpu()
        del cov_gpu
        torch.cuda.empty_cache()
    idx = torch.argsort(vals, descending=True)
    return vals[idx].float(), vecs[:, idx].float()


def eig_cached(cov, model_name, pair, variant):
    cache_path = EIGEN_CACHE_DIR / model_name / pair / f"v2_{safe_name(variant)}.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["eigenvalues"].float(), data["eigenvectors"].float()
    vals, vecs = eig_sorted(cov)
    ensure_dir(cache_path.parent)
    torch.save({"eigenvalues": vals, "eigenvectors": vecs}, cache_path)
    return vals, vecs


def cumulative_variance(eigenvalues):
    vals = eigenvalues.detach().cpu().numpy().astype(np.float64)
    denom = max(float(vals.sum()), 1e-30)
    return np.cumsum(vals) / denom


def concept_cdf(v, eigenvectors):
    v = v.detach().float()
    projections = eigenvectors.T @ v
    energy = (projections ** 2) / (v @ v).clamp(min=1e-12)
    return torch.cumsum(energy, dim=0).detach().cpu().numpy()


def gini_deviation(cdf, cumvar):
    return float(np.trapezoid(np.asarray(cdf) - np.asarray(cumvar), np.asarray(cumvar)))


def spectral_center_of_mass(cdf, cumvar):
    idx = int(np.searchsorted(cdf, 0.5))
    if idx >= len(cumvar):
        return 1.0
    return float(cumvar[idx])


def summarize_vectors(vectors, eigenvalues, eigenvectors):
    cumvar = cumulative_variance(eigenvalues)
    rows = []
    for name, v in vectors.items():
        if v.numel() != eigenvectors.shape[0] or float(v.norm()) <= 1e-12:
            continue
        cdf = concept_cdf(v, eigenvectors)
        rows.append({
            "concept": name,
            "gini": gini_deviation(cdf, cumvar),
            "scm": spectral_center_of_mass(cdf, cumvar),
            "norm": float(v.norm()),
        })
    if not rows:
        return {"n": 0}
    ginis = np.array([r["gini"] for r in rows], dtype=float)
    scms = np.array([r["scm"] for r in rows], dtype=float)
    t_stat, p_val = sp_stats.ttest_1samp(ginis, 0.0) if len(ginis) > 1 else (float("nan"), float("nan"))
    return {
        "n": len(rows),
        "mean_gini": float(ginis.mean()),
        "std_gini": float(ginis.std()),
        "mean_scm": float(scms.mean()),
        "std_scm": float(scms.std()),
        "t_stat_gini_vs_zero": float(t_stat),
        "p_gini_vs_zero": float(p_val),
        "per_concept": rows,
    }


def covariance_from_rows(rows, weights=None, center=False, batch_size=None):
    rows = rows.detach().float().cpu()
    if weights is None:
        if center:
            mean = rows.mean(dim=0, keepdim=True)
            rows = rows - mean
        return (rows.T @ rows) / max(rows.shape[0], 1)
    weights = torch.as_tensor(weights, dtype=torch.float32).cpu()
    weights = weights / weights.sum().clamp(min=1e-12)
    if center:
        mean = (rows * weights[:, None]).sum(dim=0, keepdim=True)
        rows = rows - mean
    return rows.T @ (rows * weights[:, None])


def context_token_counts(tokenizer, handler, pair="en-fr", side="src"):
    path = Path("data/paired_contexts") / f"{pair}.jsonl"
    counts = Counter()
    token_set = set()
    if not path.exists():
        return token_set, counts
    pairs = load_jsonl(path)
    for item in pairs:
        lang0 = item.get("lang0", "src")
        lang1 = item.get("lang1", "tgt")
        if side == "src":
            contexts = item.get("contexts0", [])
            src_lang, tgt_lang = lang0, lang1
        else:
            contexts = item.get("contexts1", [])
            src_lang, tgt_lang = lang1, lang0
        for text in contexts:
            formatted = handler.format_input(text, source_lang=src_lang, target_lang_code=tgt_lang, tokenizer=tokenizer)
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            counts.update(ids)
            token_set.update(ids)
    return token_set, counts


def clean_token_ids(tokenizer, vocab_size, max_tokens=None):
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids = []
    for tok_id in range(vocab_size):
        if tok_id in special:
            continue
        text = tokenizer.decode([tok_id]).strip()
        if not text:
            continue
        ids.append(tok_id)
        if max_tokens is not None and len(ids) >= max_tokens:
            break
    return ids


class LazyModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.handler = None

    def load(self):
        if self.model is None:
            print(f"  Loading model: {self.model_path}")
            self.model, self.tokenizer, self.handler = lrg.load_model(self.model_path)
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model, self.tokenizer, self.handler

    def clear(self):
        if self.model is not None:
            del self.model
        self.model = self.tokenizer = self.handler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_unembedding_rows(model):
    return lrg.get_lm_head(model).weight.detach().float().cpu()


def get_input_embedding_rows(model):
    emb = model.get_input_embeddings()
    if emb is None:
        return None
    return emb.weight.detach().float().cpu()


def run_covariance_sensitivity(args, model_name, model_path, lazy_model):
    out_dir = OUTPUT_DIR / "covariance_sensitivity" / model_name
    ensure_dir(out_dir)
    vectors = load_concept_vectors(model_name, args.pair)
    if not vectors:
        print(f"  [covariance] SKIP {model_name}: no v_raw vectors")
        return []

    rows = []
    variants = []
    try:
        saved_cov, saved_path = load_covariance(model_name, args.pair, "src")
        variants.append(("saved_language_subset_uncentered", saved_cov, {"source": str(saved_path)}))
    except Exception as e:
        print(f"  [covariance] saved covariance unavailable: {e}")

    model, tokenizer, handler = lazy_model.load()
    W_U = get_unembedding_rows(model)
    vocab_size = W_U.shape[0]
    lang_ids, lang_counts = context_token_counts(tokenizer, handler, args.pair, "src")
    lang_ids = sorted(i for i in lang_ids if 0 <= i < vocab_size)

    if not args.skip_full_vocab:
        variants.append(("full_vocab_uncentered_uniform", covariance_from_rows(W_U, center=False), {"n_tokens": len(W_U)}))
        variants.append(("full_vocab_centered_uniform", covariance_from_rows(W_U, center=True), {"n_tokens": len(W_U)}))

    if lang_ids:
        subset = W_U[lang_ids]
        weights = np.array([lang_counts.get(i, 0) for i in lang_ids], dtype=np.float64)
        if weights.sum() <= 0:
            weights = None
        variants.append(("language_subset_uncentered_uniform", covariance_from_rows(subset, center=False), {"n_tokens": len(lang_ids)}))
        variants.append(("language_subset_centered_uniform", covariance_from_rows(subset, center=True), {"n_tokens": len(lang_ids)}))
        if weights is not None:
            variants.append(("language_subset_uncentered_frequency_weighted", covariance_from_rows(subset, weights=weights, center=False), {"n_tokens": len(lang_ids)}))
            variants.append(("language_subset_centered_frequency_weighted", covariance_from_rows(subset, weights=weights, center=True), {"n_tokens": len(lang_ids)}))

    input_rows = get_input_embedding_rows(model)
    tie_info = {"has_input_embeddings": input_rows is not None}
    if input_rows is not None and input_rows.shape == W_U.shape:
        diff = (input_rows - W_U).abs()
        sample_n = min(5000, W_U.shape[0])
        idx = torch.linspace(0, W_U.shape[0] - 1, sample_n).long()
        tie_info.update({
            "same_shape": True,
            "max_abs_diff_sample": float(diff[idx].max()),
            "mean_cosine_sample": float(torch.nn.functional.cosine_similarity(input_rows[idx], W_U[idx], dim=1).mean()),
            "config_tie_word_embeddings": bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False)),
        })
        if not args.skip_embedding_spectrum:
            variants.append(("input_embedding_full_uncentered_uniform", covariance_from_rows(input_rows, center=False), {"n_tokens": len(input_rows)}))
            variants.append(("input_embedding_full_centered_uniform", covariance_from_rows(input_rows, center=True), {"n_tokens": len(input_rows)}))
    else:
        tie_info["same_shape"] = False

    for variant, cov, meta in variants:
        print(f"  [covariance] {model_name}: {variant}")
        vals, vecs = eig_cached(cov, model_name, args.pair, variant)
        summary = summarize_vectors(vectors, vals, vecs)
        row = {
            "model": model_name,
            "pair": args.pair,
            "variant": variant,
            "n_concepts": summary.get("n", 0),
            "mean_gini": summary.get("mean_gini"),
            "std_gini": summary.get("std_gini"),
            "mean_scm": summary.get("mean_scm"),
            "std_scm": summary.get("std_scm"),
            "p_gini_vs_zero": summary.get("p_gini_vs_zero"),
            "top10_variance": float(cumulative_variance(vals)[max(0, int(0.1 * len(vals)) - 1)]),
        }
        row.update(meta)
        rows.append(row)
        save_json(out_dir / f"{safe_name(variant)}.json", {"summary": summary, "metadata": meta})

    save_json(out_dir / "tied_vs_untied_embedding_info.json", tie_info)
    save_csv(out_dir / "covariance_sensitivity_summary.csv", rows)
    save_json(out_dir / "covariance_sensitivity_summary.json", rows)
    return rows


def whitening_from_cov(cov, reg_lambda):
    d = cov.shape[0]
    vals, vecs = eig_sorted(cov + reg_lambda * torch.eye(d))
    inv_sqrt = 1.0 / torch.sqrt(vals.clamp(min=1e-12))
    sqrt = torch.sqrt(vals.clamp(min=0))
    return vecs @ torch.diag(inv_sqrt) @ vecs.T, vecs @ torch.diag(sqrt) @ vecs.T


def fake_covariance_from_eigenvalues(eigenvalues, seed):
    d = len(eigenvalues)
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
    return q @ torch.diag(eigenvalues.float()) @ q.T


def run_wca_scores(cov_src, cov_tgt, X_src, X_tgt, vectors, reg_lambda):
    psi_src, _ = whitening_from_cov(cov_src, reg_lambda)
    psi_tgt, sqrt_tgt = whitening_from_cov(cov_tgt, reg_lambda)
    Q = lrg.solve_whitened_procrustes(X_src.float(), X_tgt.float(), psi_src, psi_tgt)
    out = {}
    for name, v in vectors.items():
        vt = lrg.transport_concept_vector(v.float(), psi_src, sqrt_tgt, Q)
        out[name] = float(torch.nn.functional.cosine_similarity(vt.unsqueeze(0), v.float().unsqueeze(0)))
    return out


def run_naive_scores(X_src, X_tgt, vectors):
    Q = lrg.solve_naive_procrustes(X_src.float(), X_tgt.float())
    out = {}
    for name, v in vectors.items():
        vt = lrg.transport_naive(v.float(), Q)
        out[name] = float(torch.nn.functional.cosine_similarity(vt.unsqueeze(0), v.float().unsqueeze(0)))
    return out


def run_lambda_sensitivity(args, model_name):
    out_dir = OUTPUT_DIR / "lambda_sensitivity" / model_name
    ensure_dir(out_dir)
    rows = []
    for pair in args.lambda_pairs:
        try:
            _, cov_src, cov_tgt, X_src, X_tgt, vectors = load_alignment_data(model_name, pair)
        except Exception as e:
            print(f"  [lambda] SKIP {model_name}/{pair}: {e}")
            continue
        naive = run_naive_scores(X_src, X_tgt, vectors)
        src_vals, _ = eig_sorted(cov_src)
        tgt_vals, _ = eig_sorted(cov_tgt)
        for lam in args.lambdas:
            print(f"  [lambda] {model_name}/{pair}: lambda={lam}")
            try:
                real = run_wca_scores(cov_src, cov_tgt, X_src, X_tgt, vectors, lam)
            except Exception as e:
                print(f"    real WCA failed: {e}")
                continue
            fake_seed_rows = []
            for seed in range(args.seeds):
                fake_src = fake_covariance_from_eigenvalues(src_vals, 1000 * seed + 17)
                fake_tgt = fake_covariance_from_eigenvalues(tgt_vals, 1000 * seed + 29)
                fake = run_wca_scores(fake_src, fake_tgt, X_src, X_tgt, vectors, lam)
                for concept in sorted(vectors):
                    fake_seed_rows.append({
                        "seed": seed,
                        "concept": concept,
                        "fake_score": fake[concept],
                        "fake_delta_vs_naive": fake[concept] - naive[concept],
                    })
            real_deltas = np.array([real[c] - naive[c] for c in vectors], dtype=float)
            fake_deltas = np.array([r["fake_delta_vs_naive"] for r in fake_seed_rows], dtype=float)
            fake_by_concept = defaultdict(list)
            for r in fake_seed_rows:
                fake_by_concept[r["concept"]].append(r["fake_score"])
            paired_real = []
            paired_fake = []
            for c in sorted(vectors):
                paired_real.append(real[c] - naive[c])
                paired_fake.append(np.mean(fake_by_concept[c]) - naive[c])
            t_stat, p_val = sp_stats.ttest_rel(paired_real, paired_fake) if len(paired_real) > 1 else (float("nan"), float("nan"))
            row = {
                "model": model_name,
                "pair": pair,
                "lambda": lam,
                "n_concepts": len(vectors),
                "n_seeds": args.seeds,
                "real_win_rate_vs_naive": float(np.mean(real_deltas > 0)),
                "fake_win_rate_vs_naive": float(np.mean(fake_deltas > 0)),
                "real_mean_delta_vs_naive": float(real_deltas.mean()),
                "fake_mean_delta_vs_naive": float(fake_deltas.mean()),
                "paired_t_real_minus_fake": float(t_stat),
                "paired_p_real_vs_fake": float(p_val),
            }
            rows.append(row)
            save_json(out_dir / f"lambda_{lam}_{pair}.json", {"summary": row, "fake_seed_rows": fake_seed_rows})
    save_csv(out_dir / "lambda_sensitivity_summary.csv", rows)
    save_json(out_dir / "lambda_sensitivity_summary.json", rows)
    return rows


def token_id_for_word(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    ids_sp = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    if len(ids_sp) == 1:
        return ids_sp[0]
    return None


def ids_for_concept_pairs(tokenizer, concept_name):
    out = []
    for base, target in load_word_pairs(concept_name):
        b = token_id_for_word(tokenizer, base)
        t = token_id_for_word(tokenizer, target)
        if b is not None and t is not None:
            out.append((b, t))
    return out


def unembed_pair_direction(W_U, pairs):
    if not pairs:
        return None
    diffs = [W_U[t] - W_U[b] for b, t in pairs]
    v = torch.stack(diffs).mean(dim=0)
    if float(v.norm()) <= 1e-12:
        return None
    return v / v.norm()


def sample_frequency_matched_pair(rng, all_ids, count_bins, base_id, target_id):
    def sample_like(tok_id):
        b = count_bins.get(tok_id, 0)
        candidates = [i for i in all_ids if count_bins.get(i, 0) == b and i != tok_id]
        if len(candidates) < 8:
            candidates = [i for i in all_ids if abs(count_bins.get(i, 0) - b) <= 1 and i != tok_id]
        if not candidates:
            candidates = all_ids
        return int(rng.choice(candidates))
    return sample_like(base_id), sample_like(target_id)


def run_unembedding_pair_nulls(args, model_name, lazy_model, eigenvalues, eigenvectors):
    model, tokenizer, _ = lazy_model.load()
    W_U = get_unembedding_rows(model)
    vocab_size = W_U.shape[0]
    lang_ids, counts = context_token_counts(tokenizer, lazy_model.handler, args.pair, "src")
    clean_ids = clean_token_ids(tokenizer, vocab_size, max_tokens=args.max_null_vocab_tokens)
    if lang_ids:
        clean_set = set(clean_ids)
        all_ids = sorted(clean_set | {i for i in lang_ids if i in clean_set})
    else:
        all_ids = clean_ids
    if len(all_ids) < 100:
        all_ids = clean_ids
    count_bins = {i: int(min(12, math.floor(math.log1p(counts.get(i, 0))))) for i in all_ids}

    class_pools = defaultdict(set)
    for cname in all_word_pair_concepts():
        cls = concept_class(cname)
        for b, t in ids_for_concept_pairs(tokenizer, cname):
            class_pools[cls].update([b, t])
    rng = np.random.default_rng(args.seed)
    cumvar = cumulative_variance(eigenvalues)
    rows = []
    for cname in all_word_pair_concepts():
        true_pairs = ids_for_concept_pairs(tokenizer, cname)
        if len(true_pairs) < 3:
            continue
        true_v = unembed_pair_direction(W_U, true_pairs)
        if true_v is None:
            continue
        true_gini = gini_deviation(concept_cdf(true_v, eigenvectors), cumvar)
        cls = concept_class(cname)
        pos_pool = sorted(class_pools.get(cls, [])) or all_ids
        freq_ginis = []
        pos_ginis = []
        for _ in range(args.random_samples):
            freq_pairs = [sample_frequency_matched_pair(rng, all_ids, count_bins, b, t) for b, t in true_pairs]
            v_freq = unembed_pair_direction(W_U, freq_pairs)
            if v_freq is not None:
                freq_ginis.append(gini_deviation(concept_cdf(v_freq, eigenvectors), cumvar))
            if len(pos_pool) >= 2:
                rand_pairs = [(int(rng.choice(pos_pool)), int(rng.choice(pos_pool))) for _ in true_pairs]
                v_pos = unembed_pair_direction(W_U, rand_pairs)
                if v_pos is not None:
                    pos_ginis.append(gini_deviation(concept_cdf(v_pos, eigenvectors), cumvar))
        rows.append({
            "model": model_name,
            "concept": cname,
            "n_pairs": len(true_pairs),
            "pos_proxy_class": cls,
            "true_unembed_gini": true_gini,
            "frequency_matched_null_mean_gini": float(np.mean(freq_ginis)) if freq_ginis else None,
            "frequency_matched_p_true_gt_null": float(np.mean(np.array(freq_ginis) >= true_gini)) if freq_ginis else None,
            "pos_proxy_matched_null_mean_gini": float(np.mean(pos_ginis)) if pos_ginis else None,
            "pos_proxy_matched_p_true_gt_null": float(np.mean(np.array(pos_ginis) >= true_gini)) if pos_ginis else None,
        })
    return rows


def run_norm_matched_null(args, model_name, eigenvalues, eigenvectors):
    vectors = load_concept_vectors(model_name, args.pair)
    rng = np.random.default_rng(args.seed)
    cumvar = cumulative_variance(eigenvalues)
    rows = []
    d = eigenvectors.shape[0]
    for cname, v in vectors.items():
        if float(v.norm()) <= 1e-12:
            continue
        true_gini = gini_deviation(concept_cdf(v, eigenvectors), cumvar)
        null_ginis = []
        norm = float(v.norm())
        for _ in range(args.random_samples):
            r = torch.tensor(rng.standard_normal(d), dtype=torch.float32)
            r = r / r.norm().clamp(min=1e-12) * norm
            null_ginis.append(gini_deviation(concept_cdf(r, eigenvectors), cumvar))
        null_arr = np.array(null_ginis)
        rows.append({
            "model": model_name,
            "concept": cname,
            "true_gini": true_gini,
            "norm_matched_null_mean_gini": float(null_arr.mean()),
            "norm_matched_null_std_gini": float(null_arr.std()),
            "p_null_le_true": float(np.mean(null_arr <= true_gini)),
            "p_null_ge_true": float(np.mean(null_arr >= true_gini)),
        })
    return rows


def extract_word_hidden_layers(model, tokenizer, handler, words, layer_indices=None, batch_size=8, max_length=64):
    device = next(model.parameters()).device
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs_by_word = {}
    n_layers = None
    for start in range(0, len(words), batch_size):
        batch = words[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden_states = list(out.hidden_states)
        n_layers = len(hidden_states) - 1
        if layer_indices is None:
            selected = list(range(len(hidden_states)))
        else:
            final_hs_idx = len(hidden_states) - 1
            selected = []
            for li in layer_indices:
                if li == -1:
                    selected.append(0)
                elif li >= final_hs_idx:
                    selected.append(final_hs_idx)
                else:
                    selected.append(li + 1)
        for j, word in enumerate(batch):
            token_positions = []
            for idx, (s, e) in enumerate(offsets[j].tolist()):
                if s == e == 0:
                    continue
                if e > s:
                    token_positions.append(idx)
            if not token_positions:
                token_positions = [int(inputs["attention_mask"][j].sum().item()) - 1]
            per_layer = {}
            for hs_idx in selected:
                h = hidden_states[hs_idx]
                vec = h[j, token_positions, :].mean(dim=0).detach().float().cpu()
                if hs_idx == len(hidden_states) - 1:
                    try:
                        vec = handler.apply_final_norm(model, vec.to(device)).detach().float().cpu()
                    except Exception:
                        pass
                layer_label = "embedding" if hs_idx == 0 else str(hs_idx - 1)
                per_layer[layer_label] = vec
            outputs_by_word[word] = per_layer
    return outputs_by_word, n_layers


def run_label_shuffle_null(args, model_name, lazy_model, eigenvalues, eigenvectors):
    model, tokenizer, handler = lazy_model.load()
    rng = np.random.default_rng(args.seed)
    cumvar = cumulative_variance(eigenvalues)
    rows = []
    for cname in tqdm(all_word_pair_concepts(), desc=f"label-shuffle {model_name}"):
        pairs = load_word_pairs(cname)
        if args.max_pairs_per_concept:
            pairs = pairs[:args.max_pairs_per_concept]
        if len(pairs) < 3:
            continue
        words = sorted(set([w for pair in pairs for w in pair]))
        try:
            reps, _ = extract_word_hidden_layers(model, tokenizer, handler, words, layer_indices=[999999], batch_size=args.batch_size)
        except Exception:
            reps, _ = extract_word_hidden_layers(model, tokenizer, handler, words, layer_indices=None, batch_size=args.batch_size)
        final_key = sorted(reps[words[0]].keys(), key=lambda x: -1 if x == "embedding" else int(x))[-1]
        diffs = []
        for b, t in pairs:
            if b in reps and t in reps:
                diffs.append(reps[t][final_key] - reps[b][final_key])
        if len(diffs) < 3:
            continue
        diffs = torch.stack(diffs)
        true_v = diffs.mean(dim=0)
        true_gini = gini_deviation(concept_cdf(true_v, eigenvectors), cumvar)
        sign_ginis = []
        perm_ginis = []
        base_vecs = torch.stack([reps[b][final_key] for b, _ in pairs if b in reps])
        target_vecs = torch.stack([reps[t][final_key] for _, t in pairs if t in reps])
        n = min(len(base_vecs), len(target_vecs), len(diffs))
        if n < 3:
            continue
        base_vecs = base_vecs[:n]
        target_vecs = target_vecs[:n]
        diffs = diffs[:n]
        for _ in range(args.random_samples):
            signs = torch.tensor(rng.choice([-1.0, 1.0], size=n), dtype=torch.float32)[:, None]
            v_sign = (diffs * signs).mean(dim=0)
            sign_ginis.append(gini_deviation(concept_cdf(v_sign, eigenvectors), cumvar))
            perm = torch.tensor(rng.permutation(n), dtype=torch.long)
            v_perm = (target_vecs[perm] - base_vecs).mean(dim=0)
            perm_ginis.append(gini_deviation(concept_cdf(v_perm, eigenvectors), cumvar))
        rows.append({
            "model": model_name,
            "concept": cname,
            "n_pairs": n,
            "true_activation_gini": true_gini,
            "sign_shuffle_mean_gini": float(np.mean(sign_ginis)),
            "pair_shuffle_mean_gini": float(np.mean(perm_ginis)),
            "p_sign_shuffle_le_true": float(np.mean(np.array(sign_ginis) <= true_gini)),
            "p_pair_shuffle_le_true": float(np.mean(np.array(perm_ginis) <= true_gini)),
        })
    return rows


def run_null_experiments(args, model_name, model_path, lazy_model):
    out_dir = OUTPUT_DIR / "strong_nulls" / model_name
    ensure_dir(out_dir)
    try:
        cov, _ = load_covariance(model_name, args.pair, "src")
    except Exception as e:
        print(f"  [nulls] SKIP {model_name}: {e}")
        return []
    eigenvalues, eigenvectors = eig_cached(cov, model_name, args.pair, "saved_language_subset_uncentered")
    all_rows = []
    if "norm" in args.null_types or "all" in args.null_types:
        rows = run_norm_matched_null(args, model_name, eigenvalues, eigenvectors)
        save_csv(out_dir / "norm_matched_random_directions.csv", rows)
        save_json(out_dir / "norm_matched_random_directions.json", rows)
        all_rows.extend({**r, "null_family": "norm_matched_random_directions"} for r in rows)
    if any(x in args.null_types for x in ["frequency", "pos", "all"]):
        rows = run_unembedding_pair_nulls(args, model_name, lazy_model, eigenvalues, eigenvectors)
        save_csv(out_dir / "frequency_pos_matched_unembedding_pairs.csv", rows)
        save_json(out_dir / "frequency_pos_matched_unembedding_pairs.json", rows)
        all_rows.extend({**r, "null_family": "frequency_pos_matched_unembedding_pairs"} for r in rows)
    if "shuffle" in args.null_types or "all" in args.null_types:
        rows = run_label_shuffle_null(args, model_name, lazy_model, eigenvalues, eigenvectors)
        save_csv(out_dir / "within_concept_label_shuffles.csv", rows)
        save_json(out_dir / "within_concept_label_shuffles.json", rows)
        all_rows.extend({**r, "null_family": "within_concept_label_shuffles"} for r in rows)
    save_json(out_dir / "strong_nulls_all.json", all_rows)
    return all_rows


def run_layerwise(args, model_name, lazy_model):
    out_dir = OUTPUT_DIR / "layerwise_rotation" / model_name
    ensure_dir(out_dir)
    try:
        cov, _ = load_covariance(model_name, args.pair, "src")
    except Exception as e:
        print(f"  [layerwise] SKIP {model_name}: {e}")
        return []
    eigenvalues, eigenvectors = eig_cached(cov, model_name, args.pair, "saved_language_subset_uncentered")
    cumvar = cumulative_variance(eigenvalues)
    model, tokenizer, handler = lazy_model.load()
    concepts = args.layerwise_concepts or all_word_pair_concepts()
    rows = []
    for cname in tqdm(concepts, desc=f"layerwise {model_name}"):
        pairs = load_word_pairs(cname)
        if args.max_pairs_per_concept:
            pairs = pairs[:args.max_pairs_per_concept]
        if len(pairs) < 3:
            continue
        words = sorted(set([w for pair in pairs for w in pair]))
        reps, n_layers = extract_word_hidden_layers(model, tokenizer, handler, words, layer_indices=None, batch_size=args.batch_size)
        labels = ["embedding"] + [str(i) for i in range(n_layers)]
        if args.layer_step > 1:
            labels = [labels[0]] + [lab for lab in labels[1:] if int(lab) % args.layer_step == 0 or int(lab) == n_layers - 1]
        for lab in labels:
            diffs = []
            activations = []
            for b, t in pairs:
                if b in reps and t in reps and lab in reps[b] and lab in reps[t]:
                    diffs.append(reps[t][lab] - reps[b][lab])
                    activations.extend([reps[b][lab], reps[t][lab]])
            if len(diffs) < 3:
                continue
            v = torch.stack(diffs).mean(dim=0)
            cdf = concept_cdf(v, eigenvectors)
            row = {
                "model": model_name,
                "concept": cname,
                "layer": lab,
                "layer_index": -1 if lab == "embedding" else int(lab),
                "n_pairs": len(diffs),
                "gini_final_unembed_spectrum": gini_deviation(cdf, cumvar),
                "scm_final_unembed_spectrum": spectral_center_of_mass(cdf, cumvar),
                "top10_energy_final_unembed_spectrum": float(cdf[max(0, int(0.1 * len(cdf)) - 1)]),
            }
            if args.layer_activation_spectrum and len(activations) > 3:
                A = torch.stack(activations)
                cov_layer = covariance_from_rows(A, center=True)
                vals_l, vecs_l = eig_sorted(cov_layer)
                cum_l = cumulative_variance(vals_l)
                cdf_l = concept_cdf(v, vecs_l)
                row.update({
                    "gini_layer_activation_spectrum": gini_deviation(cdf_l, cum_l),
                    "scm_layer_activation_spectrum": spectral_center_of_mass(cdf_l, cum_l),
                })
            rows.append(row)
    save_csv(out_dir / "layerwise_rotation.csv", rows)
    save_json(out_dir / "layerwise_rotation.json", rows)
    plot_layerwise(out_dir / "layerwise_rotation.png", rows, model_name)
    return rows


def plot_layerwise(path, rows, model_name):
    if not rows:
        return
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer_index"]].append(r["gini_final_unembed_spectrum"])
    xs = sorted(by_layer)
    ys = [float(np.mean(by_layer[x])) for x in xs]
    yerr = [float(np.std(by_layer[x])) for x in xs]
    plt.figure(figsize=(8, 5))
    plt.errorbar(xs, ys, yerr=yerr, marker="o", lw=2, capsize=3)
    plt.axhline(0, color="grey", ls="--", lw=1)
    plt.xlabel("Layer (-1 = embedding)")
    plt.ylabel("Mean Gini deviation vs final unembedding spectrum")
    plt.title(f"Layer-wise spectral trajectory — {model_name}")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def project_random_k(v, eigenvectors, k, rng):
    d = eigenvectors.shape[1]
    idx = rng.choice(d, size=k, replace=False)
    coeffs = eigenvectors.T @ v.float()
    mask = torch.zeros(d)
    mask[idx] = 1.0
    out = eigenvectors @ (coeffs * mask)
    if float(out.norm()) > 1e-12:
        out = out * (v.norm() / out.norm())
    return out.detach().float()


def generate_with_vector(model, tokenizer, layers, vector, alpha, layer_indices, prompt, max_new_tokens=32):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if vector is None or alpha == 0:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    else:
        with MultiLayerSteeringContext(vector, alpha, layers, layer_indices, position="all"):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def run_behavioral(args, model_name, lazy_model):
    if not all([get_layers, decompose_vector, compute_perplexity_heldout, compute_logit_shift, MultiLayerSteeringContext]):
        print("  [behavioral] SKIP: split-injection utilities unavailable")
        return []
    out_dir = OUTPUT_DIR / "behavioral_steering" / model_name
    ensure_dir(out_dir)
    vectors = load_concept_vectors(model_name, args.pair)
    if not vectors:
        print(f"  [behavioral] SKIP {model_name}: no vectors")
        return []
    cov, _ = load_covariance(model_name, args.pair, "src")
    eigenvalues, eigenvectors = eig_cached(cov, model_name, args.pair, "saved_language_subset_uncentered")
    model, tokenizer, _ = lazy_model.load()
    layers = get_layers(model)
    n_layers = len(layers)
    layer_indices = sorted(set(min(int(n_layers * f), n_layers - 1) for f in [0.50, 0.62, 0.75, 0.87]))
    heldout = load_heldout_passages(args.heldout_path)[:args.max_heldout_passages]
    concepts = args.behavioral_concepts or [c for c in vectors if c in CONCEPT_TESTS][:args.max_behavioral_concepts]
    rng = np.random.default_rng(args.seed)
    top_k = max(1, int(eigenvectors.shape[0] * args.top_k_frac))
    rows = []
    generations = []
    for cname in concepts:
        if cname not in vectors or cname not in CONCEPT_TESTS:
            continue
        v = vectors[cname]
        decomp = decompose_vector(v, eigenvectors, eigenvalues, top_k)
        components = {
            "full": v.float(),
            "top_k": decomp["v_shout"],
            "bottom_k": decomp["v_whisper"],
            "random_k": project_random_k(v, eigenvectors, top_k, rng),
        }
        base_ppls = compute_perplexity_heldout(model, tokenizer, heldout)
        base_ppl = float(np.mean(base_ppls))
        cfg = CONCEPT_TESTS[cname]
        for alpha in args.behavioral_alphas:
            for comp, vec in components.items():
                ppls = compute_perplexity_heldout(model, tokenizer, heldout, layers=layers, vector=vec, alpha=alpha, layer_indices=layer_indices)
                shifts = []
                kls = []
                for prompt in cfg["test_prompts"][:args.max_behavioral_prompts]:
                    data = compute_logit_shift(model, tokenizer, prompt, layers, vec, alpha, layer_indices, cfg["pos_keywords"], cfg["neg_keywords"])
                    shifts.append(data["concept_shift"])
                    kls.append(data["kl_divergence"])
                rows.append({
                    "model": model_name,
                    "concept": cname,
                    "component": comp,
                    "alpha": alpha,
                    "baseline_ppl": base_ppl,
                    "steered_ppl": float(np.mean(ppls)),
                    "ppl_increase_pct": float((np.mean(ppls) / base_ppl - 1.0) * 100.0) if base_ppl > 0 else None,
                    "mean_concept_shift": float(np.mean(shifts)) if shifts else 0.0,
                    "mean_kl": float(np.mean(kls)) if kls else 0.0,
                })
            if args.generate_examples:
                prompt = cfg["test_prompts"][0]
                generations.append({"concept": cname, "alpha": alpha, "component": "baseline", "prompt": prompt, "text": generate_with_vector(model, tokenizer, layers, None, 0, layer_indices, prompt, args.max_new_tokens)})
                for comp in ["full", "top_k", "bottom_k", "random_k"]:
                    generations.append({"concept": cname, "alpha": alpha, "component": comp, "prompt": prompt, "text": generate_with_vector(model, tokenizer, layers, components[comp], alpha, layer_indices, prompt, args.max_new_tokens)})
    save_csv(out_dir / "behavioral_steering.csv", rows)
    save_json(out_dir / "behavioral_steering.json", {"rows": rows, "generations": generations})
    write_behavioral_markdown(out_dir / "behavioral_steering_examples.md", rows, generations)
    return rows


def write_behavioral_markdown(path, rows, generations):
    lines = ["# Behavioral Steering Examples", ""]
    if rows:
        lines.extend(["## Quantitative summary", "", "| Concept | Component | Alpha | PPL increase % | Concept shift | KL |", "|---|---:|---:|---:|---:|---:|"])
        for r in rows:
            lines.append(f"| {r['concept']} | {r['component']} | {r['alpha']} | {r['ppl_increase_pct']:.2f} | {r['mean_concept_shift']:.6f} | {r['mean_kl']:.6f} |")
        lines.append("")
    if generations:
        lines.extend(["## Generations", ""])
        for g in generations:
            lines.extend([f"### {g['concept']} — {g['component']} — alpha={g['alpha']}", "", f"Prompt: `{g['prompt']}`", "", g["text"], ""])
    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines))


def run_sae_controls(args, model_name, lazy_model):
    try:
        import artifact_check_sae as sae_mod
    except Exception as e:
        print(f"  [sae_controls] SKIP: cannot import artifact_check_sae ({e})")
        return []
    if model_name not in getattr(sae_mod, "SAE_CONFIG", {}):
        print(f"  [sae_controls] SKIP {model_name}: no SAE config")
        return []
    out_dir = OUTPUT_DIR / "sae_frequency_controls" / model_name
    ensure_dir(out_dir)
    cfg = sae_mod.SAE_CONFIG[model_name]
    layer = args.sae_layer if args.sae_layer is not None else cfg["default_layer"]
    width = args.sae_width or cfg["default_width"]
    cov, _ = load_covariance(model_name, args.pair, "src")
    eigenvalues, eigenvectors = eig_cached(cov, model_name, args.pair, "saved_language_subset_uncentered")
    cumvar = cumulative_variance(eigenvalues)
    model, tokenizer, handler = lazy_model.load()
    device = next(model.parameters()).device
    print(f"  [sae_controls] loading SAE {model_name} layer={layer} width={width}")
    sae, method = sae_mod.load_sae(model_name, layer, width)
    sae = sae.to(device)
    sae.eval()
    concepts = all_word_pair_concepts()
    words = []
    for cname in concepts:
        for b, t in load_word_pairs(cname):
            words.extend([b, t])
    words = sorted(set(words))[:args.sae_max_words]
    reps, _ = extract_word_hidden_layers(model, tokenizer, handler, words, layer_indices=[layer], batch_size=args.batch_size)
    X = torch.stack([reps[w][str(layer)] for w in words if w in reps and str(layer) in reps[w]]).to(device)
    with torch.no_grad():
        Z = sae.encode(X) if hasattr(sae, "encode") else sae(X)[1]
    freq = (Z.detach().cpu() > 0).float().mean(dim=0).numpy()
    if hasattr(sae, "W_dec"):
        W_dec = sae.W_dec.detach().cpu().float()
    else:
        raise RuntimeError("SAE decoder matrix unavailable")
    rng = np.random.default_rng(args.seed)
    rows = []
    for cname in concepts:
        pairs = load_word_pairs(cname)
        if args.max_pairs_per_concept:
            pairs = pairs[:args.max_pairs_per_concept]
        if len(pairs) < 3:
            continue
        base = [b for b, _ in pairs]
        target = [t for _, t in pairs]
        try:
            top_indices, diff_act, decoder_vecs = sae_mod.identify_sae_concept_features(base, target, sae, model, tokenizer, handler, device, layer, top_k=args.sae_top_k)
        except Exception as e:
            print(f"    [sae_controls] {cname} failed: {e}")
            continue
        if not top_indices:
            continue
        selected_ginis = []
        matched_ginis = []
        for idx in top_indices:
            idx = int(idx)
            v = W_dec[idx]
            selected_ginis.append(gini_deviation(concept_cdf(v, eigenvectors), cumvar))
            distances = np.abs(freq - freq[idx])
            distances[idx] = np.inf
            pool = np.argsort(distances)[:max(10, args.sae_match_pool)]
            for _ in range(args.random_samples):
                ridx = int(rng.choice(pool))
                matched_ginis.append(gini_deviation(concept_cdf(W_dec[ridx], eigenvectors), cumvar))
        rows.append({
            "model": model_name,
            "concept": cname,
            "top_features": top_indices,
            "selected_mean_activation_frequency": float(np.mean([freq[int(i)] for i in top_indices])),
            "selected_mean_gini": float(np.mean(selected_ginis)),
            "activation_frequency_matched_mean_gini": float(np.mean(matched_ginis)) if matched_ginis else None,
            "p_matched_le_selected": float(np.mean(np.array(matched_ginis) <= np.mean(selected_ginis))) if matched_ginis else None,
            "sae_method": method,
            "layer": layer,
            "width": width,
        })
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_csv(out_dir / "sae_activation_frequency_controls.csv", rows)
    save_json(out_dir / "sae_activation_frequency_controls.json", rows)
    return rows


def run_for_model(args, model_arg):
    model_path = model_path_from_arg(model_arg)
    model_name = model_name_from_path(model_path)
    print("\n" + "=" * 80)
    print(f"V2 EXPERIMENTS: {model_name} ({model_path})")
    print("=" * 80)
    lazy_model = LazyModel(model_path)
    model_summary = {"model": model_name, "model_path": model_path, "experiments": {}}
    try:
        if "covariance" in args.experiments:
            model_summary["experiments"]["covariance"] = run_covariance_sensitivity(args, model_name, model_path, lazy_model)
        if "lambda" in args.experiments:
            model_summary["experiments"]["lambda"] = run_lambda_sensitivity(args, model_name)
        if "nulls" in args.experiments:
            model_summary["experiments"]["nulls"] = run_null_experiments(args, model_name, model_path, lazy_model)
        if "layerwise" in args.experiments:
            model_summary["experiments"]["layerwise"] = run_layerwise(args, model_name, lazy_model)
        if "behavioral" in args.experiments:
            model_summary["experiments"]["behavioral"] = run_behavioral(args, model_name, lazy_model)
        if "sae_controls" in args.experiments:
            model_summary["experiments"]["sae_controls"] = run_sae_controls(args, model_name, lazy_model)
    finally:
        lazy_model.clear()
    save_json(OUTPUT_DIR / model_name / "v2_experiment_summary.json", model_summary)
    return model_summary


def expand_experiments(values):
    if not values or "all" in values:
        return ALL_EXPERIMENTS
    expanded = []
    aliases = {"lambda_sensitivity": "lambda", "strong_nulls": "nulls", "sae": "sae_controls"}
    for v in values:
        v = aliases.get(v, v)
        if v not in ALL_EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{v}'. Valid: all, {', '.join(ALL_EXPERIMENTS)}")
        expanded.append(v)
    return expanded


def main():
    parser = argparse.ArgumentParser(description="Run V2 robustness/null/layerwise/behavioral experiments.")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-3B"], help="Model paths or short result names")
    parser.add_argument("--pair", default="en-fr")
    parser.add_argument("--experiments", nargs="+", default=["all"], help="all or subset: covariance lambda nulls layerwise behavioral sae_controls")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--lambda-pairs", nargs="+", default=["en-fr", "es-de", "fr-de", "fr-es"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--random-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-types", nargs="+", default=["all"], help="all, norm, frequency, pos, shuffle")
    parser.add_argument("--max-null-vocab-tokens", type=int, default=50000)
    parser.add_argument("--max-pairs-per-concept", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-full-vocab", action="store_true")
    parser.add_argument("--skip-embedding-spectrum", action="store_true")
    parser.add_argument("--layer-step", type=int, default=1)
    parser.add_argument("--layerwise-concepts", nargs="+", default=None)
    parser.add_argument("--layer-activation-spectrum", action="store_true")
    parser.add_argument("--top-k-frac", type=float, default=0.1)
    parser.add_argument("--heldout-path", default="data/heldout_ppl_passages.txt")
    parser.add_argument("--max-heldout-passages", type=int, default=5)
    parser.add_argument("--behavioral-concepts", nargs="+", default=None)
    parser.add_argument("--max-behavioral-concepts", type=int, default=3)
    parser.add_argument("--behavioral-alphas", nargs="+", type=float, default=[1.0, 5.0, 10.0])
    parser.add_argument("--max-behavioral-prompts", type=int, default=4)
    parser.add_argument("--generate-examples", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sae-layer", type=int, default=None)
    parser.add_argument("--sae-width", default=None)
    parser.add_argument("--sae-top-k", type=int, default=1)
    parser.add_argument("--sae-max-words", type=int, default=512)
    parser.add_argument("--sae-match-pool", type=int, default=200)
    args = parser.parse_args()

    args.experiments = expand_experiments(args.experiments)
    if args.max_pairs_per_concept == 0:
        args.max_pairs_per_concept = None
    ensure_dir(OUTPUT_DIR)
    print("=" * 80)
    print("V2 EXPERIMENT SUITE")
    print(f"Experiments: {args.experiments}")
    print(f"Models: {args.models}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)
    summaries = []
    for model in args.models:
        try:
            summaries.append(run_for_model(args, model))
        except Exception as e:
            print(f"ERROR while running {model}: {e}")
            import traceback
            traceback.print_exc()
            summaries.append({"model_arg": model, "error": str(e)})
    save_json(OUTPUT_DIR / "v2_experiments_all_summary.json", summaries)
    print("\nDONE")
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
