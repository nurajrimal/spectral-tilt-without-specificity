#!/usr/bin/env python3
"""
Build statement-pair datasets from both source repositories.

    geometry-of-truth   https://github.com/saprmarks/geometry-of-truth
    refusal_direction   https://github.com/andyrdt/refusal_direction

Repos are cloned to --cache-dir (default ~/data/external), outside this tree, so
no upstream data is redistributed here. Commit hashes and licences land in
provenance.json.

What gets built
---------------
Truthfulness, as separate sub-datasets rather than one pool, because the
proposal treats agreement across them as an internal replication and
disagreement as a finding about how unified the truth direction really is:

    truth_cities            748   matched    same city, true vs false country
    truth_cities_negated    748   matched    the same, with "not"
    truth_translation       354   matched    affirmed vs negated gloss
    truth_magnitude        1980   matched    larger-than vs smaller-than, same numerals
    truth_counterfact     ~7000   matched    same subject, true vs false target
    truth_conjunction      ~750   matched    "both A and B"
    truth_disjunction      ~750   matched    "either A or B"
    truth_common_claim     2225   UNMATCHED  human-labelled claims
    truth_companies         600   UNMATCHED  company facts

The first four are the proposal's named sub-datasets. The rest extend the
internal replication at no extra cost.

Refusal, which has no pairing structure available anywhere in the source repo:

    refusal                 256   UNMATCHED  the authors' own train split
    refusal_jailbreakbench  100   UNMATCHED  carries topic labels
    refusal_harmbench       159   UNMATCHED  the §4 evaluation behaviours
    refusal_advbench        520   UNMATCHED  the largest single source

Not built, deliberately
-----------------------
`likely.csv` is in the truth repo but is not a truth dataset. It contains
non-factual text ending in a probable or improbable token, and exists to prove
truth probes are not probability probes. Building it as a truth concept would
invert the control into a confound. If you want it, use it as a nuisance
contrast in Phase 2, never as a concept.

`geonames.csv` is the raw city list `cities.csv` was generated from, and
`likely_old.csv` is superseded.
"""

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPOS = {
    "geometry-of-truth": {
        "url": "https://github.com/saprmarks/geometry-of-truth.git",
        # No LICENSE file in the repo as of this writing. Adapted datasets
        # (CommonClaim, Counterfact, companies) carry their upstream terms; the
        # curated ones have no stated licence. Local research use is the
        # uncontroversial case — confirm with the author before redistributing.
        "license": "unstated (no LICENSE file; see repo)",
    },
    "refusal_direction": {
        "url": "https://github.com/andyrdt/refusal_direction.git",
        "license": "Apache-2.0",
    },
    "CAA": {
        "url": "https://github.com/nrimsky/CAA.git",
        "license": "MIT",
    },
    "evals": {
        "url": "https://github.com/anthropics/evals.git",
        "license": "CC-BY-4.0",
    },
}

PROPOSAL_PRIMARY = {"truth_cities", "truth_magnitude",
                    "truth_translation", "truth_counterfact"}

# §9 of the accepted proposal: "If a further dataset becomes available and
# passes a construct check, sycophancy is the natural addition." It is built
# here but excluded from --primary-only, because adding it is a scope decision
# for the supervisor, not a default.
PROPOSAL_OPTIONAL = {"sycophancy", "sycophancy_nlp",
                     "sycophancy_philpapers", "sycophancy_political"}


# ─── repo handling ───────────────────────────────────────────────────────────

def clone_or_find(cache: Path, name: str, do_clone: bool) -> Path:
    p = cache / name
    if p.exists():
        return p
    if not do_clone:
        raise SystemExit(f"{p} not found. Use --clone, or:\n"
                         f"  git clone {REPOS[name]['url']} {p}")
    cache.mkdir(parents=True, exist_ok=True)
    print(f"cloning {name} -> {p}")
    subprocess.run(["git", "clone", "--depth", "1", REPOS[name]["url"], str(p)],
                   check=True)
    return p


def commit(p: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(recs: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rec(concept, sub, pid, pos, neg, matched, source, **extra):
    d = {"concept": concept, "subdataset": sub, "positive": pos,
         "negative": neg, "matched": matched, "source": source}
    if pid is not None:
        d["pair_id"] = pid
    d.update(extra)
    return d


# ─── truth: pairing on the generating field ──────────────────────────────────

def _pair_on_field(rows, field, concept, sub, prefix, source, **extra_from):
    """Group rows by a generating field; emit one true/false pair per group."""
    g = defaultdict(lambda: {"1": [], "0": []})
    for r in rows:
        key = r.get(field)
        if key:
            g[key][r["label"]].append(r)
    out = []
    for i, (key, grp) in enumerate(sorted(g.items())):
        if not grp["1"] or not grp["0"]:
            continue
        p, n = grp["1"][0], grp["0"][0]
        extra = {k: p.get(v) for k, v in extra_from.items() if p.get(v)}
        out.append(rec(concept, sub, f"{prefix}_{i:05d}",
                       p["statement"], n["statement"], True, source,
                       subject=key, **extra))
    return out


def build_cities(gt: Path, negated: bool) -> list[dict]:
    name = "neg_cities" if negated else "cities"
    sub = "cities_negated" if negated else "cities"
    return _pair_on_field(read_csv(gt / "datasets" / f"{name}.csv"), "city",
                          f"truth_{sub}", sub, "cit" + ("n" if negated else ""),
                          f"geometry-of-truth/datasets/{name}.csv",
                          topic="correct_country")


def build_counterfact(gt: Path) -> list[dict]:
    """
    Counterfact ships both the true and the falsified version of each claim,
    sharing `subject` and `relation`. Pairing on (subject, relation) keeps the
    relation fixed so the contrast is truth, not relation type.
    """
    rows = read_csv(gt / "datasets" / "counterfact_true_false.csv")
    g = defaultdict(lambda: {"1": [], "0": []})
    for r in rows:
        g[(r.get("subject"), r.get("relation"))][r["label"]].append(r)
    out = []
    for i, (key, grp) in enumerate(sorted(g.items(), key=lambda kv: str(kv[0]))):
        if not grp["1"] or not grp["0"]:
            continue
        p, n = grp["1"][0], grp["0"][0]
        out.append(rec("truth_counterfact", "counterfact", f"cf_{i:05d}",
                       p["statement"], n["statement"], True,
                       "geometry-of-truth/datasets/counterfact_true_false.csv",
                       subject=p.get("subject"), topic=p.get("relation")))
    return out


_SP = re.compile(r"The Spanish word '([^']+)'")


def build_translation(gt: Path) -> list[dict]:
    """
    Each Spanish word appears once per file (344 unique in 354 rows), so there
    is no true/false pair inside sp_en_trans.csv — pairing on the word yields 5
    items. The two files are row-aligned negations, so the pair is the affirmed
    and negated form of the same gloss: one word of surface difference, truth
    value flipped.

    Asserts that negation is not confounded with truth. The original file is
    balanced 177/177, so the true member should be negated about half the time.
    If that ever stops holding the direction becomes a negation detector, which
    would still look like a clean result.
    """
    a = read_csv(gt / "datasets" / "sp_en_trans.csv")
    b = read_csv(gt / "datasets" / "neg_sp_en_trans.csv")
    if len(a) != len(b):
        raise SystemExit("sp_en_trans / neg_sp_en_trans are not row-aligned")

    out, n_neg_true = [], 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        wa, wb = _SP.search(ra["statement"]), _SP.search(rb["statement"])
        if not wa or not wb or wa.group(1) != wb.group(1) or ra["label"] == rb["label"]:
            continue
        t, f = (ra, rb) if ra["label"] == "1" else (rb, ra)
        if t is rb:
            n_neg_true += 1
        out.append(rec("truth_translation", "translation", f"spen_{i:05d}",
                       t["statement"], f["statement"], True,
                       "geometry-of-truth/datasets/{sp_en_trans,neg_sp_en_trans}.csv",
                       subject=wa.group(1), true_member_negated=(t is rb)))

    if out:
        frac = n_neg_true / len(out)
        if not 0.35 <= frac <= 0.65:
            raise SystemExit(f"truth_translation: true member negated in {frac:.0%} "
                             f"of pairs — negation is confounded with truth here.")
        print(f"    (true member negated in {frac:.0%} of pairs — balanced)")
    return out


def build_magnitude(gt: Path) -> list[dict]:
    """
    Pair larger_than against smaller_than on the identical numerals. "X is
    larger than Y" and "X is smaller than Y" share both numbers and differ in
    one word, which is tighter than pairing inside either file alone.
    """
    lt = {(r["n1"], r["n2"]): r for r in read_csv(gt / "datasets" / "larger_than.csv")}
    st = {(r["n1"], r["n2"]): r for r in read_csv(gt / "datasets" / "smaller_than.csv")}
    out = []
    for i, k in enumerate(sorted(lt.keys() & st.keys())):
        a, b = lt[k], st[k]
        if a["label"] == b["label"]:
            continue
        p, n = (a, b) if a["label"] == "1" else (b, a)
        out.append(rec("truth_magnitude", "magnitude", f"mag_{i:05d}",
                       p["statement"], n["statement"], True,
                       "geometry-of-truth/datasets/{larger,smaller}_than.csv",
                       topic="magnitude", abs_diff=int(a["abs_diff"])))
    return out


def build_logical(gt: Path, disj: bool) -> list[dict]:
    """
    Conjunctions and disjunctions of city statements. No generating field links
    a true to a false form, so pairs are formed by index within the file: both
    members share the template and the construction, differing in truth value.
    Structurally matched, not minimally matched — flagged as such in the record.
    """
    name = "cities_cities_disj" if disj else "cities_cities_conj"
    sub = "disjunction" if disj else "conjunction"
    rows = read_csv(gt / "datasets" / f"{name}.csv")
    t = [r for r in rows if r["label"] == "1"]
    f = [r for r in rows if r["label"] == "0"]
    n = min(len(t), len(f))
    return [rec(f"truth_{sub}", sub, f"{sub[:4]}_{i:05d}",
                t[i]["statement"], f[i]["statement"], True,
                f"geometry-of-truth/datasets/{name}.csv",
                pairing="structural_not_minimal")
            for i in range(n)]


def build_unpaired_truth(gt: Path, which: str) -> list[dict]:
    """
    CommonClaim and the companies set: true and false statements about
    different subjects, with nothing linking them. Emitted unmatched.
    """
    cfg = {"common_claim": ("common_claim_true_false.csv", "truth_common_claim"),
           "companies": ("companies_true_false.csv", "truth_companies")}[which]
    rows = read_csv(gt / "datasets" / cfg[0])
    t = [r for r in rows if r["label"] == "1"]
    f = [r for r in rows if r["label"] == "0"]
    n = min(len(t), len(f))
    return [rec(cfg[1], which, None, t[i]["statement"], f[i]["statement"],
                False, f"geometry-of-truth/datasets/{cfg[0]}")
            for i in range(n)]


# ─── refusal: no pairing available ───────────────────────────────────────────

def _refusal_from(harmful: list[dict], harmless: list[dict], concept: str,
                  sub: str, source: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = min(n, len(harmful), len(harmless))
    hf = rng.sample(harmful, n)
    hl = rng.sample(harmless, n)
    return [rec(concept, sub, None, a["instruction"], b["instruction"], False,
                source, topic=a.get("category"))
            for a, b in zip(hf, hl)]


def build_refusal(rd: Path, n: int = 256, seed: int = 0) -> list[dict]:
    """
    The authors' own train split. This is the reason to take refusal from the
    repo rather than from AdvBench upstream: these are the exact filtered,
    non-overlapping sets the paper used, and they cannot be reconstructed from
    the source corpora.

    Arditi's own config uses n_train=128. 256 is the full harmful_train pool
    (260) and gives margin; the published direction is found with half this.

    No pair_id. `positive` and `negative` on a line come from unrelated corpora
    and are aligned only by index. Extract with the unmatched estimator and read
    the label-shuffle null before believing any gap.
    """
    return _refusal_from(
        json.load(open(rd / "dataset/splits/harmful_train.json")),
        json.load(open(rd / "dataset/splits/harmless_train.json")),
        "refusal", "arditi_train",
        "refusal_direction/dataset/splits/harm{ful,less}_train.json", n, seed)


def build_refusal_source(rd: Path, which: str, n: int = 200,
                         seed: int = 0) -> list[dict]:
    """
    Each harmful corpus separately, against Alpaca controls.

    Running the sources apart is a cheap diagnostic: if the spectral placement
    moves between AdvBench, JailbreakBench and HarmBench, the direction is
    tracking corpus style rather than refusal, and that must be known before the
    headline is interpreted.
    """
    cfg = {"jailbreakbench": ("jailbreakbench.json", "refusal_jailbreakbench"),
           "harmbench": ("harmbench_test.json", "refusal_harmbench"),
           "advbench": ("advbench.json", "refusal_advbench")}[which]
    harmful = json.load(open(rd / "dataset/processed" / cfg[0]))
    harmless = json.load(open(rd / "dataset/processed/alpaca.json"))
    return _refusal_from(harmful, harmless, cfg[1], which,
                         f"refusal_direction/dataset/processed/{{{cfg[0]},alpaca.json}}",
                         n, seed)


# ─── sycophancy: an answer-token contrast ────────────────────────────────────

def _syco_records(items, concept, sub, prefix, source, limit, seed):
    """
    Build positive/negative from one prompt and two answers.

    The contrast here is not two prompts. Each item is a single question with
    the user's identity and opinion embedded, plus two answer options, one of
    which agrees with the stated opinion. Positive and negative are therefore
    the same question with a different answer appended — they differ by a
    single token.

    That makes this the tightest pairing in the study, tighter than truth's
    same-city-different-country and far tighter than refusal's unmatched
    corpora. It also changes where the direction lives: at the ANSWER token, not
    at the end of a statement. Records carry contrast_type="answer_token" so the
    extractor can warn when a pooling scheme is inappropriate for them.

    Answer letters are near-balanced upstream (489/511 in CAA), so "(A)" is not
    a proxy for the behaviour. The builder asserts this rather than trusting it.
    """
    rng = random.Random(seed)
    if limit and len(items) > limit:
        items = rng.sample(items, limit)

    def one(v):
        """
        answer_not_matching_behavior is a string in two of the three evals
        files and a list in philpapers2020, where an item can have several
        non-matching options. Take the first: with two options it is the only
        one, and with more, picking one keeps the pair a single-token contrast
        rather than an average over alternatives.
        """
        if isinstance(v, (list, tuple)):
            return str(v[0]).strip() if v else ""
        return str(v or "").strip()

    out, n_a = [], 0
    for i, r in enumerate(items):
        q = r.get("question")
        a = one(r.get("answer_matching_behavior"))
        b = one(r.get("answer_not_matching_behavior"))
        if not q or not a or not b or a == b:
            continue
        if a.startswith("(A)"):
            n_a += 1
        out.append(rec(concept, sub, f"{prefix}_{i:05d}",
                       f"{q.rstrip()}\n{a}", f"{q.rstrip()}\n{b}",
                       True, source, contrast_type="answer_token"))

    if out:
        frac = n_a / len(out)
        if not 0.35 <= frac <= 0.65:
            raise SystemExit(
                f"{concept}: the sycophantic answer is '(A)' in {frac:.0%} of "
                f"items. Answer letter is confounded with the behaviour, so the "
                f"direction would be an A-vs-B detector. Rebalance first.")
        print(f"    (sycophantic answer is '(A)' in {frac:.0%} of items — balanced)")
    return out


def build_sycophancy_caa(caa: Path, limit: int = 1000, seed: int = 0) -> list[dict]:
    """
    The CAA generation set: the 1,000 items Panickssery et al. built their
    steering vectors from. Derived from anthropics/evals, so it overlaps with
    the sets below; use one or the other for a headline, not both pooled.
    """
    items = json.load(open(caa / "datasets/generate/sycophancy/generate_dataset.json"))
    return _syco_records(items, "sycophancy", "caa_generate", "syc",
                         "CAA/datasets/generate/sycophancy/generate_dataset.json",
                         limit, seed)


def build_sycophancy_evals(ev: Path, which: str, limit: int = 500,
                           seed: int = 0) -> list[dict]:
    """
    Anthropic's model-written evaluations, kept as three separate sub-datasets.

    Same logic as the truth sub-datasets: agreement across NLP-survey,
    PhilPapers and political-typology is an internal replication, and
    disagreement says the sycophancy direction is topic-dependent. The three
    cover very different subject matter, so this is a real test rather than a
    formality.
    """
    f = {"nlp": "sycophancy_on_nlp_survey.jsonl",
         "philpapers": "sycophancy_on_philpapers2020.jsonl",
         "political": "sycophancy_on_political_typology_quiz.jsonl"}[which]
    items = [json.loads(l) for l in open(ev / "sycophancy" / f, encoding="utf-8")
             if l.strip()]
    return _syco_records(items, f"sycophancy_{which}", which, f"syc{which[:3]}",
                         f"evals/sycophancy/{f}", limit, seed)


BUILDERS = {
    "truth_cities":            ("geometry-of-truth", lambda p: build_cities(p, False)),
    "truth_cities_negated":    ("geometry-of-truth", lambda p: build_cities(p, True)),
    "truth_translation":       ("geometry-of-truth", build_translation),
    "truth_magnitude":         ("geometry-of-truth", build_magnitude),
    "truth_counterfact":       ("geometry-of-truth", build_counterfact),
    "truth_conjunction":       ("geometry-of-truth", lambda p: build_logical(p, False)),
    "truth_disjunction":       ("geometry-of-truth", lambda p: build_logical(p, True)),
    "truth_common_claim":      ("geometry-of-truth", lambda p: build_unpaired_truth(p, "common_claim")),
    "truth_companies":         ("geometry-of-truth", lambda p: build_unpaired_truth(p, "companies")),
    "refusal":                 ("refusal_direction", build_refusal),
    "refusal_jailbreakbench":  ("refusal_direction", lambda p: build_refusal_source(p, "jailbreakbench")),
    "refusal_harmbench":       ("refusal_direction", lambda p: build_refusal_source(p, "harmbench")),
    "refusal_advbench":        ("refusal_direction", lambda p: build_refusal_source(p, "advbench")),
    "sycophancy":              ("CAA", build_sycophancy_caa),
    "sycophancy_nlp":          ("evals", lambda p: build_sycophancy_evals(p, "nlp")),
    "sycophancy_philpapers":   ("evals", lambda p: build_sycophancy_evals(p, "philpapers")),
    "sycophancy_political":    ("evals", lambda p: build_sycophancy_evals(p, "political")),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(Path.home() / "data" / "external"))
    ap.add_argument("--out-dir", default="data/statements")
    ap.add_argument("--clone", action="store_true")
    ap.add_argument("--concepts", nargs="+", default=sorted(BUILDERS),
                    choices=sorted(BUILDERS))
    ap.add_argument("--primary-only", action="store_true",
                    help="only the proposal's four named truth sub-datasets plus refusal")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    concepts = args.concepts
    if args.primary_only:
        concepts = sorted(PROPOSAL_PRIMARY | {"refusal"})

    cache, out_dir = Path(args.cache_dir).expanduser(), Path(args.out_dir)
    repos = {n: clone_or_find(cache, n, args.clone)
             for n in sorted({BUILDERS[c][0] for c in concepts})}

    prov = {"generated_by": "scripts/build_datasets.py", "seed": args.seed,
            "sources": {n: {"url": REPOS[n]["url"], "license": REPOS[n]["license"],
                            "commit": commit(p), "local_path": str(p)}
                        for n, p in repos.items()},
            "outputs": {}}

    rng = random.Random(args.seed)
    print()
    n_matched = n_unmatched = 0
    for c in concepts:
        repo, fn = BUILDERS[c]
        recs = fn(repos[repo])
        if args.max_pairs and len(recs) > args.max_pairs:
            recs = rng.sample(recs, args.max_pairs)
        if not recs:
            print(f"  {c:26s} EMPTY — skipped")
            continue

        path = out_dir / f"{c}.jsonl"
        write_jsonl(recs, path)
        matched = recs[0]["matched"]
        n_matched += len(recs) if matched else 0
        n_unmatched += 0 if matched else len(recs)
        star = (" *" if c in PROPOSAL_PRIMARY
                else " +" if c in PROPOSAL_OPTIONAL else "  ")
        print(f"  {c:26s}{star} n={len(recs):6d}  "
              f"{'matched' if matched else 'UNMATCHED':9s} -> {path}")
        prov["outputs"][c] = {"path": str(path), "n_pairs": len(recs),
                              "matched": matched, "source_repo": repo,
                              "proposal_primary": c in PROPOSAL_PRIMARY}

    with open(out_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)

    print(f"\n  {n_matched + n_unmatched} pairs total "
          f"({n_matched} matched, {n_unmatched} unmatched)")
    print(f"  * = named in the accepted proposal   "
          f"+ = §9 optional extension (sycophancy)")
    print(f"  provenance -> {out_dir / 'provenance.json'}")
    print("\nMatched sets use the per-pair estimator; unmatched sets use pooled "
          "difference-of-means\nand their gap is only read alongside the "
          "label-shuffle null.")


if __name__ == "__main__":
    main()
