"""End-to-end test: synthetic model with planted structure, all phases."""
import sys, types, json, tempfile, warnings
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from linear_rep_geometry.spectral_metrics import (
    eigendecompose, compute_cumulative_variance, compute_concept_energy_cdf,
    compute_gini_deviation, compute_spectral_center_of_mass, metrics,
    subspace_metrics, random_direction_baseline, gini_gap, empirical_pvalue,
    one_sample_test, paired_test, tost_equivalence, SESOI)
from linear_rep_geometry import behavioral as beh
from linear_rep_geometry import nulls as nullmod
from linear_rep_geometry import dimensionality as dim_mod
from linear_rep_geometry import dual_geometry as dual
from linear_rep_geometry import covariance as cov_mod

torch.manual_seed(0)
D, NL, V = 64, 5, 300

Q, _ = torch.linalg.qr(torch.randn(D, D))
EV = torch.tensor([1.0 / (i + 1) ** 1.3 for i in range(D)])
COV = Q @ torch.diag(EV) @ Q.T
COV = ((COV + COV.T) / 2).float()
TAIL, HEAD = Q[:, -5], Q[:, 0]
CHOL = torch.linalg.cholesky(COV + 1e-5 * torch.eye(D))

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond: ok = False
def banner(s): print(f"\n{'='*70}\n{s}\n{'='*70}")


class Tok:
    pad_token = "<pad>"; eos_token = "<pad>"; chat_template = None
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None,
                 max_length=None, **kw):
        if isinstance(texts, str): texts = [texts]
        lens = [max(3, min(10, 3 + len(t) % 7)) for t in texts]
        T = max(lens)
        ids = torch.zeros(len(texts), T, dtype=torch.long)
        am = torch.zeros(len(texts), T, dtype=torch.long)
        for i, L in enumerate(lens):
            ids[i, :L] = torch.randint(1, V, (L,)); am[i, :L] = 1
        ns = types.SimpleNamespace(input_ids=ids, attention_mask=am)
        ns.to = lambda d: ns
        return ns
    def encode(self, t, add_special_tokens=False):
        return [abs(hash(t)) % V, (abs(hash(t)) // 7) % V]
    def decode(self, ids): return "x" * (1 + int(ids[0]) % 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
        self.flag = False; self.W = torch.randn(V, D) * 0.1
        self.lm_head = types.SimpleNamespace(weight=self.W)
    def get_output_embeddings(self): return types.SimpleNamespace(weight=self.W)
    def forward(self, ids, attention_mask=None, output_hidden_states=True):
        B, T = ids.shape
        hs = []
        for _ in range(NL + 1):
            h = torch.randn(B, T, D) @ CHOL.T
            if self.flag: h = h + 2.5 * TAIL.view(1, 1, D)
            hs.append(h)
        return types.SimpleNamespace(hidden_states=tuple(hs))


class Handler:
    def _detect_device(self, m): return "cpu"
    def apply_final_norm(self, m, h): return h
    def get_unembedding(self, m): return m.W


# 1 ── metrics
banner("1. spectral_metrics")
evals, evecs = eigendecompose(COV)
cumvar = compute_cumulative_variance(evals)
gT = metrics(TAIL, evecs, cumvar); gH = metrics(HEAD, evecs, cumvar)
check("tail -> negative Gini", gT["gini"] < 0, f"{gT['gini']:+.4f}")
check("head -> positive Gini", gH["gini"] > 0, f"{gH['gini']:+.4f}")
check("SCM orders tail>head", gT["scm"] > gH["scm"], f"{gT['scm']:.2f}>{gH['scm']:.2f}")
check("scale invariant", abs(metrics(TAIL*9.1, evecs, cumvar)["gini"] - gT["gini"]) < 1e-6)
check("sign invariant", abs(metrics(-TAIL, evecs, cumvar)["gini"] - gT["gini"]) < 1e-6)

sub = subspace_metrics(torch.stack([Q[:, -5], Q[:, -6], Q[:, -7]]), evecs, cumvar)
check("subspace of tail -> negative", sub["gini"] < 0, f"{sub['gini']:+.4f} rank={sub['rank']}")
subH = subspace_metrics(torch.stack([Q[:, 0], Q[:, 1]]), evecs, cumvar)
check("subspace of head -> positive", subH["gini"] > 0, f"{subH['gini']:+.4f}")

rnd = random_direction_baseline(D, evecs, cumvar, n_samples=200, seed=1)
check("random baseline", "gini_mean" in rnd, f"{rnd['gini_mean']:+.4f}")
check("gini_gap", abs(gini_gap(gT["gini"], rnd["gini_samples"])
                      - (gT["gini"] - rnd["gini_mean"])) < 1e-9)

# 2 ── inference
banner("2. inference: H0, paired, TOST")
strong = [-0.20, -0.18, -0.22, -0.19, -0.21, -0.17, -0.23, -0.20]
t = one_sample_test(strong)
check("H0 rejected for real effect", t["p_t"] < 0.05, f"p={t['p_t']:.2e}")
check("CI excludes zero", t["ci95"][1] < 0, f"[{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}]")
check("wilcoxon reported at n=8", "p_wilcoxon" in t)

tiny = [0.002, -0.003, 0.001, 0.000, -0.002, 0.003, -0.001, 0.002]
t2 = one_sample_test(tiny)
eq = tost_equivalence(tiny, sesoi=SESOI)
check("H0 not rejected for null effect", t2["p_t"] > 0.05, f"p={t2['p_t']:.3f}")
check("TOST declares equivalence", eq["equivalent"], f"p_tost={eq['p_tost']:.2e}")
# Genuinely inconclusive: mean sits near the bound AND the spread is wide
# enough that n=8 cannot separate it from either zero or the bound. Note how
# hard this is to construct — with n=8 and a bound of 0.082, TOST usually
# resolves, which is good news for the thesis's power at eight models.
mid = [-0.15, 0.05, -0.20, 0.10, -0.12, 0.02, -0.18, 0.08]
eq2 = tost_equivalence(mid, sesoi=SESOI)
t3 = one_sample_test(mid)
check("inconclusive case: H0 not rejected", t3["p_t"] > 0.05, f"p={t3['p_t']:.3f}")
check("inconclusive case: TOST also fails", not eq2["equivalent"],
      f"p_tost={eq2['p_tost']:.3f} -> correctly reports underpowered")

pt = paired_test([-0.05]*4+[-0.04]*4, [-0.20]*4+[-0.19]*4, "refusal", "truth")
check("paired test separates classes", pt["p_t"] < 0.05, f"p={pt['p_t']:.2e}")
try:
    paired_test([1,2,3], [1,2]); check("paired shape guard", False)
except ValueError: check("paired shape guard", True)

# 3 ── extraction
banner("3. behavioral extraction")
pairs = [beh.StatementPair("c", f"pos statement number {i}", f"neg statement number {i}",
                           pair_id=f"p{i}", matched=True) for i in range(24)]
mdl, tok, hnd = Model(), Tok(), Handler()
calls = {"n": 0}; orig = mdl.forward
def fwd(ids, attention_mask=None, output_hidden_states=True):
    mdl.flag = calls["n"] < 3; calls["n"] += 1
    return orig(ids, attention_mask, output_hidden_states)
mdl.forward = fwd

res = beh.extract_concept(pairs, mdl, tok, hnd, device="cpu", batch_size=8)
exp = len(beh.POOLING) * (NL + 1)
check(f"{exp} directions", len(res["directions"]) == exp, str(len(res["directions"])))
check("diffs retained", len(res["diffs"]) == exp)
check("diff shape [n,d]", tuple(res["diffs"][("last_token", 0)].shape) == (24, D))
check("unit norm", all(abs(float(v.norm())-1) < 1e-4 for v in res["directions"].values()))
check("both poolings present", set(res["acts"]) == set(beh.POOLING))
check("recovers planted tail",
      abs(float(torch.dot(res["directions"][("last_token", 2)], TAIL))) > 0.5,
      f"|cos|={abs(float(torch.dot(res['directions'][('last_token',2)], TAIL))):.2f}")

with tempfile.TemporaryDirectory() as td:
    beh.save_concept(res, td, "c", save_acts=True)
    rl = beh.load_concept(Path(td)/"concept_c.pt")
    check("save/load roundtrip", set(rl["directions"]) == set(res["directions"]))
    k = ("last_token", 1)
    check("diffs roundtrip", bool(torch.allclose(rl["diffs"][k], res["diffs"][k])))

# 4 ── nulls
banner("4. nulls (Phase 2)")
xp = torch.randn(120, D) @ CHOL.T + 2.0*TAIL
xn = torch.randn(120, D) @ CHOL.T
sh = nullmod.label_shuffle_null(xp, xn, evecs, cumvar, n_perm=200, seed=0)
check("shuffle detects real signal", sh["p_two_sided"] < 0.05,
      f"gap={sh['gap']:+.4f} p={sh['p_two_sided']:.4f}")
xp0 = torch.randn(120, D) @ CHOL.T; xn0 = torch.randn(120, D) @ CHOL.T
sh0 = nullmod.label_shuffle_null(xp0, xn0, evecs, cumvar, n_perm=200, seed=0)
check("shuffle correctly null on no-signal", sh0["p_two_sided"] > 0.05,
      f"p={sh0['p_two_sided']:.4f}")
shm = nullmod.label_shuffle_null(xp, xn, evecs, cumvar, n_perm=200, seed=0, matched=True)
check("matched sign-flip variant runs", "gap" in shm, f"gap={shm['gap']:+.4f}")

bn = nullmod.both_nulls(xp, xn, evecs, cumvar, n_perm=200, n_random=200, seed=0)
check("both nulls reported", "random_null" in bn and "shuffle_null" in bn)
check("agreement flag present", isinstance(bn["agree"], bool), f"agree={bn['agree']}")

rel = nullmod.split_half_reliability(xp, xn)
check("split-half high on signal", rel["cosine_mean"] > 0.7, f"{rel['cosine_mean']:.2f}")
rel0 = nullmod.split_half_reliability(xp0, xn0)
check("split-half low on noise", rel0["cosine_mean"] < 0.5, f"{rel0['cosine_mean']:.2f}")
mv = nullmod.matched_vs_unmatched(xp, xn, evecs, cumvar)
check("matched vs unmatched", "cosine" in mv, f"cos={mv['cosine']:.2f}")

with tempfile.TemporaryDirectory() as td:
    p = Path(td)/"rule.json"
    nullmod.freeze_decision_rule(p, nullmod.DEFAULT_RULE)
    check("rule written with hash", json.load(open(p)).get("sha256") is not None)
    try:
        nullmod.freeze_decision_rule(p, nullmod.DEFAULT_RULE); check("write-once", False)
    except FileExistsError: check("rule is write-once", True)

# 5 ── dimensionality
banner("5. dimensionality (Phase 4)")
D1 = (torch.randn(100, 1) @ TAIL.unsqueeze(0)) * 3 + torch.randn(100, D)*0.05
r1 = dim_mod.analyze_rank(D1)
check("rank-one detected", r1["is_rank_one"] or r1["cos_pc1_mean"] > 0.9,
      f"PC1={r1['pc1_var_explained']:.2f} cos={r1['cos_pc1_mean']:.2f}")
check("participation ratio low", r1["participation_ratio"] < 10,
      f"PR={r1['participation_ratio']:.2f}")
D3 = sum((torch.randn(100,1) @ Q[:, -(5+i)].unsqueeze(0)) for i in range(3)) \
     + torch.randn(100, D)*0.05
r3 = dim_mod.analyze_rank(D3)
check("multi-dim flagged", not r3["is_rank_one"], f"PC1={r3['pc1_var_explained']:.2f}")
check("PR reflects rank 3", 2.0 < r3["participation_ratio"] < 5.0,
      f"PR={r3['participation_ratio']:.2f}")
bs = dim_mod.concept_subspace(D3, 3)
check("subspace basis shape", tuple(bs.shape) == (3, D))
sm = subspace_metrics(bs, evecs, cumvar)
check("subspace of tail dirs -> negative", sm["gini"] < 0, f"{sm['gini']:+.4f}")
rep = dim_mod.rank_report({("last_token", i): D1 for i in range(3)})
check("rank_report summarises", rep["summary"]["treat_as_rank_one"])

# 6 ── covariance bases
banner("6. covariance (Phase 5)")
cfull = cov_mod.unembedding_covariance(mdl, hnd, None, lam=0.1)
check("full-vocab cov shape", tuple(cfull.shape) == (D, D))
check("symmetric", bool(torch.allclose(cfull, cfull.T, atol=1e-5)))
check("positive definite", bool(torch.linalg.eigvalsh(cfull).min() > 0))
csub = cov_mod.unembedding_covariance(mdl, hnd, list(range(50)), lam=0.1)
check("language-subset cov differs", not torch.allclose(cfull, csub, atol=1e-3))
cact = cov_mod.activation_covariance([f"text {i}" for i in range(40)], mdl, tok, hnd,
                                     layer=-1, device="cpu", batch_size=8, lam=0.1)
check("activation cov PD", bool(torch.linalg.eigvalsh(cact).min() > 0))

# 7 ── dual geometry
banner("7. dual geometry (Phase 6)")
vc = dual.build_vocabulary_contrast(mdl, tok, hnd)
check("both tokenisation variants built", set(vc) == {"plain", "leading_space"})
usable = [v for v in vc.values() if v.get("ok")]
check("at least one variant usable", len(usable) > 0)
if usable:
    check("balance reported", "char_len_balanced" in usable[0])
dg = dual.dual_geometry_test(mdl, tok, hnd, -0.25, evecs, cumvar,
                             n_samples=100, seed=0)
check("verdict produced", "sign_flip" in dg["verdict"] or "note" in dg["verdict"])
check("variant agreement checked", "variants_agree_on_sign" in dg["verdict"]
      or not any(r.get("ok") for r in dg["variants"].values()))
ntn = dual.random_token_pair_null(mdl, hnd, 5, evecs, cumvar, n_samples=100)
check("token-pair null runs", "gini_mean" in ntn, f"{ntn['gini_mean']:+.4f}")

print(f"\n{'='*70}\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}\n{'='*70}")
sys.exit(0 if ok else 1)
