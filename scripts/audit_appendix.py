#!/usr/bin/env python3
"""Audit paper appendix tables against saved result artifacts.

Compares the numbers PRINTED IN main.tex against the JSONs that produced them,
flagging any mismatch beyond tolerance. Run on the box holding results/_spectral.
"""
import json, os, statistics as st

SP = "results/_spectral"
def load(p): return json.load(open(os.path.join(SP, p)))
def flag(name, paper, got, tol):
    if got is None: return f"  [MISSING] {name}: paper={paper}"
    d = abs(paper - got)
    mark = "OK " if d <= tol else "**FLAG**"
    return f"  {mark} {name:34} paper={paper:>8}  got={got:>8.3f}  Δ={d:.3f}"

# ── Paper values from main.tex ──────────────────────────────────────────────
# Table tab:full-scm  (model -> (concept_scm, gini))
SPECTRAL = {
 "Qwen2.5-0.5B":(0.997,-0.356),"Qwen2.5-1.5B":(0.986,-0.282),"Qwen2.5-3B":(0.973,-0.272),
 "Qwen2.5-7B":(0.999,-0.396),"Qwen2.5-14B":(0.996,-0.346),"Qwen3-0.6B":(0.938,-0.216),
 "Qwen3-1.7B":(0.997,-0.248),"Qwen3-4B":(0.994,-0.312),"Qwen3-8B":(1.000,-0.359),
 "Llama-3.2-1B":(0.738,-0.140),"Llama-3.2-3B":(0.771,-0.159),"gemma-2-2b":(0.964,-0.389),
 "gemma-2-9b":(0.927,-0.364),"Mistral-7B-v0.3":(0.908,-0.199),"SmolLM2-1.7B":(1.000,-0.436),
 "OLMoE-1B-7B-0125":(0.620,-0.076),"jetmoe-8b":(0.945,-0.242),
}
SPECTRAL_MEAN = (0.926,-0.282)
# Table tab:sae-check  (label -> mean gini)
SAE = {"gemma-2-2b(top1)":-0.230, "gemma-2-9b":-0.276, "Llama-3.1-8B":-0.193}
# Table tab:pos-probe-full  (model -> (topk, botk, gap))
POS = {
 "gemma-2-2b":(.554,.506,.048),"Llama-3.2-3B":(.628,.576,.052),"Llama-3.1-8B":(.641,.625,.016),
 "Mistral-7B-v0.3":(.624,.607,.017),"SmolLM2-1.7B":(.560,.536,.024),"Qwen3-4B":(.572,.555,.018),
 "Qwen2.5-3B":(.557,.576,-.019),"Qwen2.5-7B":(.572,.586,-.014),
}
# Table tab:dual-geometry  (model -> (null_gini, concept_gini))
DUAL = {"Qwen2.5-0.5B":(-0.095,0.046),"Llama-3.2-1B":(-0.124,0.103),
        "gemma-2-2b":(-0.117,0.073),"SmolLM2-1.7B":(-0.071,0.056)}

print("="*70); print("1) SPECTRAL METRICS TABLE (17 models: Concept SCM, Gini)"); print("="*70)
try:
    d = load("experiment1_summary.json")
    by = {m["model"]: m for m in d["models"]}
    scms, ginis = [], []
    for model,(pscm,pgini) in SPECTRAL.items():
        m = by.get(model)
        if not m: print(f"  [MISSING MODEL] {model}"); continue
        scms.append(m["mean_scm"]); ginis.append(m["mean_gini"])
        print(flag(f"{model} SCM", pscm, m["mean_scm"], 0.003))
        print(flag(f"{model} Gini", pgini, m["mean_gini"], 0.003))
    print(flag("MEAN SCM (all 17)", SPECTRAL_MEAN[0], st.mean(scms), 0.003))
    print(flag("MEAN Gini (all 17)", SPECTRAL_MEAN[1], st.mean(ginis), 0.003))
except Exception as e: print("  ERROR:", e)

print("\n"+"="*70); print("2) SAE ARTIFACT-CHECK TABLE (mean Gini)"); print("="*70)
try:
    for label,fn in [("gemma-2-2b(top1)","artifact_check/artifact_check_sae_gemma-2-2b.json"),
                     ("gemma-2-9b","artifact_check/artifact_check_sae_gemma-2-9b.json"),
                     ("Llama-3.1-8B","artifact_check/artifact_check_sae_Llama-3.1-8B.json")]:
        try:
            d = load(fn)
            ginis = [v["gini_deviation"] for v in d.values() if isinstance(v,dict) and "gini_deviation" in v]
            print(flag(f"{label} mean Gini (n={len(ginis)})", SAE[label], st.mean(ginis), 0.01))
        except Exception as e: print(f"  [{label}] err:", e)
except Exception as e: print("  ERROR:", e)

print("\n"+"="*70); print("3) POS PROBING TABLE (top-k, bot-k)"); print("="*70)
for model,(ptop,pbot,pgap) in POS.items():
    fn = f"pos_probe/pos_probe_{model}.json"
    try:
        d = load(fn)
        rec = d[0] if isinstance(d,list) else d
        # find top/bottom accuracy keys
        top = rec.get("top_k_accuracy", rec.get("top_acc"))
        bot = rec.get("bottom_k_accuracy", rec.get("bot_acc"))
        print(flag(f"{model} top-k", ptop, top, 0.004))
        print(flag(f"{model} bot-k", pbot, bot, 0.004))
    except Exception as e:
        print(f"  [{model}] err: {e}")

print("\n"+"="*70); print("4) DUAL-GEOMETRY NULL TABLE"); print("="*70)
try:
    d = load("hole_fixes/hole5_unembed_null.json")
    by = {r["model"]: r for r in d}
    for model,(pn,pc) in DUAL.items():
        r = by.get(model)
        if not r: print(f"  [MISSING] {model}"); continue
        print(flag(f"{model} null Gini", pn, r["null_gini_mean"], 0.006))
        print(flag(f"{model} concept Gini", pc, r["concept_gini_mean"], 0.006))
except Exception as e: print("  ERROR:", e)

print("\n"+"="*70); print("5) SPLIT-INJECTION (schema dump — verify manually)"); print("="*70)
try:
    d = load("hole_fixes/hole4_v2_split_injection_gemma-2-2b.json")
    r = d.get("results")
    print("  gemma-2-2b results type:", type(r).__name__,
          "| top-level keys:", list(d.keys()))
    print("  alphas:", d.get("alphas"), "n_concepts:", d.get("n_concepts"),
          "n_heldout:", d.get("n_heldout_passages"))
except Exception as e: print("  ERROR:", e)
