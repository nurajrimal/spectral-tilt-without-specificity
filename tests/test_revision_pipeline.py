#!/usr/bin/env python3
"""
End-to-end check of the revision code paths, with no model, GPU or download.

Plants a small synthetic world, runs both probe scripts through their real
run functions (window sweep, all controls, MLP probe, length modes, folder
guard), then runs the two-tail report on the outputs. It checks plumbing and
invariants, not effect sizes.

    python tests/test_revision_pipeline.py          # ~2-4 min on a CPU
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check(cond, msg):
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def make_cov(path, d, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) * np.geomspace(5, 0.05, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(A @ A.T / d, dtype=torch.float32), path)


def syntax_world(tmp, model_name, d=48, n_sent=150, seed=0):
    rng = np.random.default_rng(seed)
    tags = ["NOUN", "VERB", "ADJ", "DET", "ADP"]
    vocab = [f"w{i}" for i in range(120)]
    tag_of = {w: tags[i % len(tags)] for i, w in enumerate(vocab)}
    emb = {w: rng.standard_normal(d) for w in vocab}
    tag_vec = {t: rng.standard_normal(d) * 2 for t in tags}
    sents, words, ts, acts = [], [], [], []
    for _ in range(n_sent):
        ws = list(rng.choice(vocab, size=8))
        sents.append([(w, tag_of[w]) for w in ws])
        for w in ws:
            words.append(w)
            ts.append(tag_of[w])
            acts.append(emb[w] + tag_vec[tag_of[w]] + rng.standard_normal(d) * 0.5)
    acts_dir = tmp / "acts_syntax"
    acts_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": torch.tensor(np.array(acts), dtype=torch.float32),
                "tags": ts, "words": words}, acts_dir / f"acts_{model_name}_500sent.pt")

    class Pub:
        @staticmethod
        def download_ud_english():
            return "unused"

        @staticmethod
        def parse_conllu(_):
            return sents

        @staticmethod
        def project_activations(a, e, idx):
            return (a.float() @ e[:, idx].float()).numpy()
    return Pub(), acts_dir


def syntax_args(**kw):
    a = dict(lang_pair="en-fr", k_frac=0.10, n_windows=4, n_random=2, n_boot=300,
             max_sentences=500, control="token_id_random", ud_path=None,
             published_json=None, require_gpu=False, refresh_acts=False,
             no_figure=True, layer="final", k_fracs=[0.05, 0.25], force=False,
             cov=None)
    a.update(kw)
    return argparse.Namespace(**a)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="revision_test_"))
    print(f"workdir {tmp}")
    tsc = load("test_syntax_claim", REPO / "scripts" / "test_syntax_claim.py")
    tsc.PROBE_GPU = False

    # ── loud tail ───────────────────────────────────────────────────────────
    print("\nloud tail")
    models = ["FakeA", "FakeB", "FakeC"]
    fv_dir = tmp / "loud_fullvocab"
    for i, m in enumerate(models):
        pub, acts_dir = syntax_world(tmp, m, seed=i)
        cov = tmp / "results" / m / "cov" / "cov_full_vocab.pt"
        make_cov(cov, 48, seed=i)
        tsc.OUT_DIR, tsc.ACTS_DIR, tsc.PROBE_KIND = fv_dir, acts_dir, "linear"
        out = tsc.run_model(f"org/{m}", syntax_args(cov=str(cov)), pub)
    check(set(out["did_by_control"]) == set(tsc.CONTROLS), "DiD against all three controls")
    check(set(out["by_k"]) == {"0.1", "0.05", "0.25"}, "window sweep stored per k")
    check(out["basis_kind"] == "full_vocab" and out["basis_sha256"], "basis provenance recorded")
    check(not tsc.audit_folder(fv_dir, verbose=False), "clean folder passes the audit")

    # The bug that produced p = 0.10: a published-basis file in a full-vocab folder.
    pub, acts_dir = syntax_world(tmp, "FakeD", seed=9)
    pcov = tmp / "results" / "FakeD" / "en-fr" / "cov_src.pt"
    make_cov(pcov, 48, seed=9)
    tsc.ACTS_DIR = acts_dir
    try:
        tsc.run_model("org/FakeD", syntax_args(cov=str(pcov)), pub)
        refused = False
    except SystemExit:
        refused = True
    check(refused, "folder guard refuses a published-basis file in a full-vocab folder")
    shutil.copy(REPO / "results/_spectral/syntax_claim/gemma-2-2b.json", fv_dir)
    check(bool(tsc.audit_folder(fv_dir, verbose=False)), "audit flags a copied-in file")
    (fv_dir / "gemma-2-2b.json").unlink()

    # Published-basis folder for the same fake models, for the report.
    pb_dir = tmp / "loud_published"
    for i, m in enumerate(models):
        pub, acts_dir = syntax_world(tmp, m, seed=i)
        pcov = tmp / "results" / m / "en-fr" / "cov_src.pt"
        make_cov(pcov, 48, seed=100 + i)
        tsc.OUT_DIR, tsc.ACTS_DIR = pb_dir, acts_dir
        tsc.run_model(f"org/{m}", syntax_args(cov=str(pcov)), pub)

    # MLP on one model.
    mlp_dir = tmp / "loud_mlp"
    pub, acts_dir = syntax_world(tmp, "FakeA", seed=0)
    tsc.OUT_DIR, tsc.ACTS_DIR, tsc.PROBE_KIND = mlp_dir, acts_dir, "mlp"
    out = tsc.run_model("org/FakeA", syntax_args(
        cov=str(tmp / "results/FakeA/en-fr/cov_src.pt"), k_fracs=None), pub)
    check(out["probe"] == "mlp" and out["full_accuracy"]["pos"] > 0.5,
          f"MLP probe runs (full-d POS acc {out['full_accuracy']['pos']:.2f})")
    tsc.PROBE_KIND = "linear"

    # ── quiet tail ──────────────────────────────────────────────────────────
    print("\nquiet tail")
    cs = load("test_concept_subspace", REPO / "scripts" / "test_concept_subspace.py")
    stmts = tmp / "statements"
    stmts.mkdir()
    cs.STATEMENTS = stmts
    cs.depth_from_config = lambda _m: 5            # roles: mid=2, final=4
    rng = np.random.default_rng(1)
    n_pairs, d = 240, 48
    for concept, len_bias in (("truth_fake", 0.0), ("truth_longbias", 0.35)):
        with open(stmts / f"{concept}.jsonl", "w") as f:
            for i in range(n_pairs):
                base = "x" * int(rng.integers(20, 60))
                extra_pos = int(rng.integers(0, 6)) + (5 if rng.random() < len_bias else 0)
                extra_neg = int(rng.integers(0, 6))
                f.write(json.dumps({"concept": concept,
                                    "positive": base + "a" * extra_pos + " yes",
                                    "negative": base + "b" * extra_neg + " no",
                                    "matched": True}) + "\n")
    q_dir, q_acts = tmp / "quiet", tmp / "acts_quiet"
    cs.ACTS_DIR = q_acts
    q_acts.mkdir()
    from linear_rep_geometry.behavioral import load_pairs
    for m_i, m in enumerate(models):
        for concept in ("truth_fake", "truth_longbias"):
            pairs = load_pairs(stmts / f"{concept}.jsonl")
            lens = np.array([len(p.positive) for p in pairs] +
                            [len(p.negative) for p in pairs], dtype=float)
            item = rng.standard_normal((n_pairs, d)) * np.geomspace(5, 0.05, d)
            X = np.r_[item, item] + rng.standard_normal((2 * n_pairs, d)) * 0.05
            X[:n_pairs, -4:] += 1.0
            X[:, 20:23] += 3 * ((lens - lens.mean()) / lens.std())[:, None]
            A = torch.tensor(X, dtype=torch.float32)
            torch.save({"acts": {2: A, 4: A * 1.1}, "n_layers": 5, "d": d},
                       q_acts / f"acts_{m}__{concept}__last_token__L2-4.pt")
            cov = tmp / "results" / m / "en-fr" / "cov_src.pt"
            for mode, odir in (("none", q_dir), ("balance", tmp / "quiet_balance"),
                               ("residualise", tmp / "quiet_resid")):
                cs.OUT_DIR = odir
                a = argparse.Namespace(
                    max_pairs=512, control="length_bucket", cov=str(cov),
                    layers=["mid", "final"], pooling="last_token", n_windows=4,
                    n_boot=300, k_frac=0.05, k_fracs=[0.1, 0.25], batch_size=8,
                    max_length=256, refresh_acts=False, require_gpu=False,
                    length_mode=mode, force=False)
                r = cs.run_model_concept(tsc, f"org/{m}", concept, a)
    check(set(r["by_layer"]["2"]["by_k"]) == {"0.05", "0.1", "0.25"},
          "quiet tail sweep stored per k")
    check("chars" in r["length_diagnostics"], "length diagnostics recorded")
    rb = json.load(open(tmp / "quiet_balance" / "FakeC__truth_longbias.json"))
    lb = rb["length_diagnostics"]["chars"]
    check(abs(lb["frac_pairs_pos_longer"] - lb["frac_pairs_neg_longer"]) < 1e-9,
          f"balanced subset: positive longer as often as negative "
          f"({lb['frac_pairs_pos_longer']:.2f} vs {lb['frac_pairs_neg_longer']:.2f})")
    rows = cs.collect_rows(q_dir)
    check(len(rows) == len(models) * 2 * 2 * 3, f"collect_rows found {len(rows)} cells")

    # ── semantic control (pooled sycophancy subsets, topic) ─────────────────
    print("\nsemantic control")
    sem = load("semantic_control", REPO / "scripts" / "semantic_control.py")
    subsets = sem.SUBSETS
    for si, sub in enumerate(subsets):
        with open(stmts / f"{sub}.jsonl", "w") as f:
            for i in range(120):
                q = f"topic{si} question {i} " + "x" * int(rng.integers(10, 40))
                f.write(json.dumps({"concept": sub, "positive": q + " (A)",
                                    "negative": q + " (B)", "matched": True,
                                    "contrast_type": "answer_token"}) + "\n")
    sem_dir = tmp / "quiet_semantic"
    cs.OUT_DIR = sem_dir
    for m in models:
        for si, sub in enumerate(subsets):
            item = rng.standard_normal((120, d)) * np.geomspace(5, 0.05, d)
            item[:, 10:13] += 2.0 * si                      # topic is decodable
            X = np.r_[item, item] + rng.standard_normal((240, d)) * 0.05
            X[:120, -4:] += 1.0                             # concept, quiet end
            A = torch.tensor(X, dtype=torch.float32)
            torch.save({"acts": {2: A, 4: A}, "n_layers": 5, "d": d},
                       q_acts / f"acts_{m}__{sub}__last_token__L2-4.pt")
        a = argparse.Namespace(
            cov=str(tmp / "results" / m / "en-fr" / "cov_src.pt"),
            layers=["mid", "final"], pooling="last_token", k_frac=0.05,
            k_fracs=[0.1], n_windows=4, n_boot=300, max_pairs=512, batch_size=8,
            max_length=256, refresh_acts=False, require_gpu=False,
            skip_existing=False, control="topic")
        sem.run_model(cs, tsc, f"org/{m}", a)
    rs = json.load(open(sem_dir / "FakeA__sycophancy_pooled.json"))
    cell = rs["by_layer"]["2"]
    check(rs["n_pairs"] == 360 and cell["difference_in_differences"].get("available"),
          "semantic control: 3 subsets pooled, Delta computed")
    check(cell["headroom"]["topic"] > 0.3,
          f"topic is learnable under pair-grouped folds (headroom "
          f"{cell['headroom']['topic']:.2f})")
    check(abs(cell["full_accuracy"]["pair_id_random"] - 0.5) < 0.08,
          "pair_id_random still at chance (no leakage)")
    check(len(cs.collect_rows(sem_dir)) == len(models) * 2 * 2,
          "semantic JSON readable by the report")

    # ── report ──────────────────────────────────────────────────────────────
    print("\ntwo-tail report")
    prereg = tmp / "prereg.json"
    pr = json.load(open(REPO / "prereg" / "two_tail_prereg.json"))
    pr["loud_tail"]["primary"]["k_frac"] = 0.10
    pr["quiet_tail"]["primary"]["k_frac"] = 0.05
    pr["k_sweep"] = [0.05, 0.10, 0.25]
    pr["models"] = models
    json.dump(pr, open(prereg, "w"), indent=2)
    subprocess.run([sys.executable, str(REPO / "scripts" / "freeze_prereg.py"),
                    "--prereg", str(prereg), "--no-git"], check=True)
    plc = tmp / "placement.csv"
    with open(plc, "w") as f:
        f.write("concept,model,shuffle\n")
        for m in models:
            for c in ("truth_fake", "truth_longbias"):
                f.write(f"{c},{m},{rng.normal(-0.02, 0.02):.4f}\n")
    rep = tmp / "report"
    cmd = [sys.executable, str(REPO / "scripts" / "two_tail_report.py"),
           "--prereg", str(prereg), "--loud-final", str(pb_dir),
           "--quiet", str(q_dir), "--placement", str(plc),
           "--length-balanced", str(tmp / "quiet_balance"),
           "--length-residualised", str(tmp / "quiet_resid"),
           "--loud-mlp", str(mlp_dir),
           "--loud-fullvocab", str(fv_dir), "--quiet-semantic", str(sem_dir),
           "--out", str(rep)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-3000:])
    if res.returncode:
        print(res.stderr[-3000:])
    check(res.returncode == 0, "report runs")
    for f in ("fig1_loud_retention.pdf", "fig2_quiet_retention.pdf",
              "fig3_per_model_delta.pdf", "fig4_placement_vs_capacity.pdf",
              "figS1_k_sweep.pdf", "primary_and_family.csv", "report.md",
              "table_two_tail.tex"):
        check((rep / f).exists(), f"report wrote {f}")
    print(f"\nALL PASS  ({tmp})")


if __name__ == "__main__":
    main()
