#!/usr/bin/env python3
"""
Check the cluster environment before the first real run.

    python scripts/check_env.py
    python scripts/check_env.py --models Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-3B

Checks Python, the libraries, CUDA, disk, and access to gated model
repositories. Everything it tests is something that fails hours into a job if
left to chance: a CUDA/driver mismatch, a missing Hugging Face licence
acceptance, or a scratch quota too small for the weights.

Exits non-zero if anything required is missing, so it can gate a job script.
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# (import name, pip name, minimum, why)
REQUIRED = [
    ("torch",        "torch",         (2, 9),  "models, eigendecomposition, SVD"),
    ("numpy",        "numpy",         (2, 0),  "metrics; np.trapezoid needs 2.0+"),
    ("scipy",        "scipy",         (1, 11), "t-tests, Wilcoxon, TOST"),
    ("transformers", "transformers",  (4, 50), "model and tokenizer loading"),
    ("accelerate",   "accelerate",    (1, 0),  "device_map='auto' sharding"),
    ("safetensors",  "safetensors",   (0, 4),  "weight loading"),
    ("tqdm",         "tqdm",          (4, 60), "progress bars"),
]

OPTIONAL = [
    ("sklearn",    "scikit-learn", "published artifact_check_probe.py"),
    ("matplotlib", "matplotlib",   "figures"),
    ("seaborn",    "seaborn",      "figures"),
    ("datasets",   "datasets",     "only if you stream a corpus for the activation basis"),
]

# Approximate bf16 footprint, GB. Weights only; add ~10% for cache overhead.
MODEL_SIZES = {
    "Qwen2.5-0.5B": 1, "Qwen2.5-3B": 6, "Qwen2.5-7B": 15, "Qwen3-4B": 8,
    "Llama-3.2-3B": 6, "Llama-3.1-8B": 16, "gemma-2-2b": 5, "Mistral-7B-v0.3": 15,
}

GATED = {
    "meta-llama": "accept the Llama licence on the model page while signed in",
    "google":     "accept the Gemma terms on the model page while signed in",
    "mistralai":  "some Mistral repos require accepting terms",
}

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B",
    "google/gemma-2-2b", "mistralai/Mistral-7B-v0.3",
]

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
problems, warnings_ = [], []


def line(status, msg, detail=""):
    print(f"[{status}] {msg}" + (f"\n         {detail}" if detail else ""))


def parse_version(v: str) -> tuple:
    out = []
    for part in v.split(".")[:3]:
        num = "".join(c for c in part if c.isdigit())
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def check_python():
    v = sys.version_info
    if v >= (3, 12):
        line(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    elif v >= (3, 10):
        line(WARN, f"Python {v.major}.{v.minor}",
             "the published repo targets 3.12; 3.10+ should work")
        warnings_.append("python < 3.12")
    else:
        line(BAD, f"Python {v.major}.{v.minor}", "need 3.10 or newer")
        problems.append("python")


def check_libraries():
    for mod, pip_name, minimum, why in REQUIRED:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            line(BAD, f"{pip_name} missing", f"needed for {why}")
            problems.append(pip_name)
            continue
        ver = getattr(m, "__version__", "?")
        if ver != "?" and parse_version(ver) < minimum:
            want = ".".join(str(x) for x in minimum)
            line(WARN, f"{pip_name} {ver}", f"recommend >= {want} ({why})")
            warnings_.append(f"{pip_name} {ver}")
        else:
            line(OK, f"{pip_name} {ver}")

    for mod, pip_name, why in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            line(OK, f"{pip_name} {getattr(m, '__version__', '?')} (optional)")
        except ImportError:
            line(WARN, f"{pip_name} not installed (optional)", why)


def check_numpy_trapezoid():
    """np.trapz was renamed in 2.0; the metrics fall back, but confirm one works."""
    try:
        import numpy as np
        fn = getattr(np, "trapezoid", None) or np.trapz
        val = float(fn(np.array([0.0, 1.0]), np.array([0.0, 1.0])))
        line(OK, f"integration available ({fn.__name__}), sanity {val:.2f}")
    except Exception as e:
        line(BAD, "numpy integration unavailable", str(e))
        problems.append("numpy.trapezoid")


def check_cuda():
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        line(WARN, "CUDA not available",
             "fine on a login node; the job script must request a GPU. "
             "If this is already a GPU node, the torch build does not match "
             "the driver — reinstall torch for the cluster's CUDA version.")
        warnings_.append("no CUDA")
        return

    n = torch.cuda.device_count()
    line(OK, f"CUDA {torch.version.cuda}, {n} device(s)")
    smallest = None
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1e9
        smallest = gb if smallest is None else min(smallest, gb)
        print(f"         gpu{i}: {p.name}, {gb:.0f} GB")

    if smallest is not None:
        if smallest < 20:
            line(WARN, f"{smallest:.0f} GB per GPU",
                 "8B models in bf16 need ~16 GB of weights plus activations. "
                 "Use --batch-size 2 and --max-length 128, or stay under 4B.")
            warnings_.append("small GPU")
        elif smallest < 40:
            line(OK, f"{smallest:.0f} GB per GPU",
                 "enough for the whole suite; drop --batch-size for the 8B models")
        else:
            line(OK, f"{smallest:.0f} GB per GPU", "comfortable for all eight models")

    # Compute capability vs the torch build. A Blackwell card (sm_120) with a
    # torch wheel built for CUDA < 12.8 imports fine and then fails at the first
    # kernel launch, or silently produces nothing. Catch it here rather than
    # forty minutes into a run.
    try:
        cc = torch.cuda.get_device_capability(0)
        arches = torch.cuda.get_arch_list()
        tag = f"sm_{cc[0]}{cc[1]}"
        if any(tag in a for a in arches):
            line(OK, f"compute capability {tag} supported by this torch build")
        else:
            line(BAD, f"torch has no kernels for {tag}",
                 f"built for: {', '.join(arches)}. Reinstall with a newer CUDA "
                 f"wheel, e.g. --index-url https://download.pytorch.org/whl/cu128")
            problems.append("torch arch mismatch")
    except Exception as e:
        line(WARN, "could not read compute capability", str(e)[:100])

    try:
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x.T).sum().item()
        torch.linalg.eigh(x @ x.T + torch.eye(512, device="cuda"))
        line(OK, f"GPU matmul and eigendecomposition work (checksum {y:.1f})")
    except Exception as e:
        msg = str(e)[:140]
        line(BAD, "GPU linear algebra failed", msg)
        if "no kernel image" in msg:
            line(BAD, "", "this is the arch mismatch above — install a cu128+ wheel")
        problems.append("cuda linalg")


def check_disk(models):
    need = sum(MODEL_SIZES.get(Path(m).name, 8) for m in models)
    need = int(need * 1.1)

    cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    cache_path = Path(cache) if cache else Path.home() / ".cache" / "huggingface"
    if not cache:
        line(WARN, f"HF_HOME unset, defaulting to {cache_path}",
             "home directories are usually quota-limited on clusters. "
             "Set HF_HOME to scratch before downloading anything.")
        warnings_.append("HF_HOME unset")

    probe = cache_path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free / 1e9
    except OSError as e:
        line(WARN, f"could not stat {probe}", str(e))
        return

    msg = f"{free:.0f} GB free at {probe}; weights need ~{need} GB"
    if free < need:
        line(BAD, msg, "not enough room for the model suite")
        problems.append("disk")
    elif free < need * 1.5:
        line(WARN, msg, "tight once intermediates are written")
        warnings_.append("disk tight")
    else:
        line(OK, msg)

    try:
        res_free = shutil.disk_usage(Path.cwd()).free / 1e9
        note = ("activations are the large output: roughly n_pairs x layers x d "
                "x 4 bytes per concept. Use --no-save-acts for the breadth sweep.")
        if res_free < 50:
            line(WARN, f"{res_free:.0f} GB free in the working directory", note)
            warnings_.append("results disk")
        else:
            line(OK, f"{res_free:.0f} GB free in the working directory", note)
    except OSError:
        pass


def check_hf_access(models, probe_network):
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN"))
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    gated = sorted({m.split("/")[0] for m in models if m.split("/")[0] in GATED})

    if token or token_file.exists():
        line(OK, "Hugging Face credential found")
    elif gated:
        line(BAD, "no Hugging Face token",
             f"gated: {', '.join(gated)}. Run `huggingface-cli login`, and "
             f"accept each licence on the model page while signed in.")
        problems.append("hf token")
    else:
        line(OK, "no gated models requested; token not needed")

    for org in gated:
        line(WARN, f"{org}: gated", GATED[org] +
             " — a token alone is not enough, the licence must be accepted.")

    if not probe_network:
        return
    try:
        from huggingface_hub import model_info
        for m in models:
            try:
                model_info(m)
                line(OK, f"reachable: {m}")
            except Exception as e:
                short = str(e).split("\n")[0][:90]
                line(BAD, f"cannot access {m}", short)
                problems.append(m)
    except ImportError:
        line(WARN, "huggingface_hub not importable; skipped the access probe")


def check_repo_layout():
    for rel, why, required in [
        ("src/linear_rep_geometry", "the package", True),
        ("scripts/run_phases.py", "phase runner", True),
        ("data/word_pairs", "the §6 word-level control", True),
        ("data/statements", "built by scripts/build_datasets.py", False),
    ]:
        p = Path(rel)
        if p.exists():
            extra = ""
            if rel == "data/word_pairs":
                extra = f"{len(list(p.glob('*.txt')))} concept files"
            elif rel == "data/statements":
                extra = f"{len(list(p.glob('*.jsonl')))} datasets"
            line(OK, rel, extra)
        elif required:
            line(BAD, f"{rel} missing", why)
            problems.append(rel)
        else:
            line(WARN, f"{rel} missing", why)


def check_scheduler():
    for cmd, name in [("sinfo", "Slurm"), ("qstat", "PBS/SGE"), ("bjobs", "LSF")]:
        if shutil.which(cmd):
            line(OK, f"{name} detected ({cmd})",
                 "the published Makefile assumes an interactive GPU box; "
                 "replace that layer with a job script and leave the analysis "
                 "code untouched (proposal §9).")
            return
    line(WARN, "no scheduler found", "probably an interactive node")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--probe-network", action="store_true",
                    help="contact the Hub to confirm each model is reachable")
    args = ap.parse_args()

    print("=" * 68)
    print("Environment check")
    print("=" * 68)

    for title, fn in [
        ("Python", check_python),
        ("Libraries", check_libraries),
        ("Numerics", check_numpy_trapezoid),
        ("GPU", check_cuda),
        ("Disk", lambda: check_disk(args.models)),
        ("Model access", lambda: check_hf_access(args.models, args.probe_network)),
        ("Repository", check_repo_layout),
        ("Scheduler", check_scheduler),
    ]:
        print(f"\n--- {title} ---")
        fn()

    print("\n" + "=" * 68)
    if problems:
        print(f"{len(problems)} blocking issue(s): {', '.join(problems)}")
        print("\nInstall what is missing with:")
        print("  pip install -r requirements.txt")
        print("Torch must match the cluster's CUDA version — check the module "
              "system\nbefore installing it from PyPI.")
    elif warnings_:
        print(f"Ready, with {len(warnings_)} warning(s): {', '.join(warnings_)}")
    else:
        print("Ready.")
    print("=" * 68)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
