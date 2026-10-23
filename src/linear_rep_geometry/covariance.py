"""
Phase 5 — the spectral basis.

The published work measures against Sigma_L, the unembedding second moment taken
over a language-restricted token subset. That restriction exists because their
concepts include Language Pairs, where the relevant vocabulary is a subset. A
behavioural concept has no such subset: refusal is not a property of a token
list, so there is nothing principled to restrict to.

Three bases, and the proposal commits to reporting which conclusions move:

  full_vocab   Sigma over the whole unembedding matrix. The primary basis here.
               The paper's own Appendix D shows the result barely moves between
               full-vocabulary and language-subset, so this costs little and
               removes an arbitrary choice.

  language     The published restriction, reproduced so Phase 0 can check
               against the reported -0.356 and so the word-level control numbers
               are computed exactly as published.

  activation   Second moment of residual-stream activations over a text corpus.
               The authors flag this as unexplored and it is arguably the right
               basis for directions that never surface as a single token: a
               refusal direction lives in activation space and is never read off
               as a vocabulary row.

A result that holds under all three is basis-independent. A result that flips
between the unembedding bases and the activation basis is the more interesting
outcome, and it is a finding about where the concept lives rather than a
problem with the measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


def _unembedding_matrix(model, handler) -> torch.Tensor:
    """[vocab, d] unembedding rows. Tied-weight models return the embedding."""
    for getter in (
        lambda: handler.get_unembedding(model),
        lambda: model.get_output_embeddings().weight,
        lambda: model.lm_head.weight,
    ):
        try:
            W = getter()
            if W is not None:
                return W.detach().float().cpu()
        except (AttributeError, NotImplementedError):
            continue
    raise RuntimeError("could not locate the unembedding matrix for this model")


def unembedding_covariance(model, handler,
                           token_ids: Optional[Sequence[int]] = None,
                           lam: float = 0.0,
                           center: bool = False,
                           weights: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Sigma_L = (1/|L|) sum_w gamma(w) gamma(w)^T + lambda I.   (paper eq. 1)

    token_ids=None gives the full-vocabulary basis. Passing the published
    language subset reproduces Sigma_L exactly.

    center=False matches the paper's main text. The centered variant is one of
    its Appendix D robustness rows and is kept here for the same purpose.
    """
    W = _unembedding_matrix(model, handler)
    if token_ids is not None:
        idx = torch.tensor(sorted(set(int(t) for t in token_ids)), dtype=torch.long)
        W = W[idx]

    if center:
        W = W - W.mean(dim=0, keepdim=True)

    if weights is not None:
        w = torch.tensor(np.asarray(weights, dtype=np.float32))
        if len(w) != len(W):
            raise ValueError(f"weights length {len(w)} != rows {len(W)}")
        w = w / w.sum()
        cov = (W * w.unsqueeze(1)).T @ W
    else:
        cov = (W.T @ W) / W.shape[0]

    cov = cov + lam * torch.eye(cov.shape[0])
    return (cov + cov.T) / 2


@torch.inference_mode()
def activation_covariance(texts: Sequence[str], model, tokenizer, handler,
                          layer: int = -1, device: str = "cuda",
                          batch_size: int = 8, max_length: int = 256,
                          max_tokens: int = 200_000, lam: float = 0.0,
                          center: bool = True, seed: int = 0) -> torch.Tensor:
    """
    Second moment of residual-stream activations at one layer, over real text.

    Accumulated as X^T X over non-padding token positions, streamed so the full
    activation matrix is never held. max_tokens caps the accumulation; 200k
    tokens is far above the point where a d x d second moment stabilises for
    d <= 5120, and the estimate is checked for conditioning before use.

    center=True by default, unlike the unembedding bases. An activation cloud
    has a large mean offset — the anisotropy literature is largely about that
    offset — and leaving it in would make the top eigenvector approximately the
    mean activation for every model, which is not a basis worth measuring
    against.

    Report the layer. "The activation covariance" is not well defined without
    one, and the spectrum changes with depth.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = handler._detect_device(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total, d = None, None
    n_tokens = 0
    mean_acc = None

    for i in range(0, len(texts), batch_size):
        if n_tokens >= max_tokens:
            break
        batch = [t for t in texts[i: i + batch_size] if t and t.strip()]
        if not batch:
            continue

        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        out = model(enc.input_ids, attention_mask=enc.attention_mask,
                    output_hidden_states=True)
        hs = out.hidden_states
        li = layer if layer >= 0 else len(hs) + layer
        h = hs[li]
        if li == len(hs) - 1:
            h = handler.apply_final_norm(model, h)

        mask = enc.attention_mask.bool()
        X = h[mask].float().cpu()                   # [n_tok, d]
        if X.numel() == 0:
            continue

        if total is None:
            d = X.shape[1]
            total = torch.zeros(d, d, dtype=torch.float64)
            mean_acc = torch.zeros(d, dtype=torch.float64)

        Xd = X.double()
        total += Xd.T @ Xd
        mean_acc += Xd.sum(dim=0)
        n_tokens += X.shape[0]
        del out, hs, h, X, Xd

    if total is None or n_tokens == 0:
        raise RuntimeError("no activations collected; check the corpus and tokenizer")

    second = total / n_tokens
    if center:
        mu = (mean_acc / n_tokens).unsqueeze(1)
        second = second - mu @ mu.T

    cov = second.float() + lam * torch.eye(d)
    cov = (cov + cov.T) / 2

    evals = torch.linalg.eigvalsh(cov)
    cond = float(evals.max() / evals.min().clamp(min=1e-12))
    print(f"    activation cov: layer {layer}, {n_tokens} tokens, "
          f"d={d}, cond={cond:.2e}")
    if cond > 1e8:
        print("    warning: ill-conditioned; raise lam or collect more tokens")
    return cov


def load_or_build(path: Path, builder, force: bool = False) -> torch.Tensor:
    """Cache a covariance to disk. These are expensive and fully deterministic."""
    path = Path(path)
    if path.exists() and not force:
        return torch.load(path, map_location="cpu", weights_only=True).float()
    cov = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cov, path)
    return cov
