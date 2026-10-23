# Revision runbook

What to run on `sprout`, in order, to produce every number and figure the
revised paper needs. Each step maps to an item on the revision list.

## 0. Before anything (10 min)

```bash
cd ~/spectral_anti_concentration
git checkout -b revision
# copy the new/changed files from this bundle into the repo, then:
.venv/bin/python tests/test_revision_pipeline.py     # synthetic end-to-end test; must end ALL PASS
bash scripts/run_revision.sh check                   # data files, disk, GPU, cached activations
```

**Disk.** New activation caches total roughly 2–3 GB (quiet tail: ~7–34 MB per
model × concept; loud mid layer: ~25–110 MB per model). You have 16 GB free.
Do not pass `--refresh-acts`: every run reuses the activations already cached
in `results/_spectral/syntax_claim/` and `results/_spectral/concept_subspace/`.

**GPU.** The machine is shared. Check `nvidia-smi` before each extraction stage
(3 and 5); 8B models need ~17 GB of VRAM.

## 1. Time one cell, then freeze the pre-registration (items 2.3–2.5)

```bash
bash scripts/run_revision.sh smoke      # reruns two cells already in the thesis
```

The smoke Δ must match the old values it prints, to about the third decimal. If
it doesn't, stop: the environment has changed since the thesis runs.

Then open `prereg/two_tail_prereg.json` and confirm, with your supervisor if
possible:

- **Primaries.** Loud: Δ vs `token_id_random`, final layer, k = 10% (the
  published setting). Quiet: Δ vs length, **mid** layer, k = 0.5% (your
  original setting).
- **SESOI = 0.10 retention units**, with the rationale written in the file.
  Choose this *now*, from the rationale, not from any result. On the existing
  two concepts the quiet-tail 90% CI is about ±0.09–0.11 wide, so 0.10 is
  borderline at n = 7–8 models. More concepts per model is what tightens it.
- **The 9 new concepts and the k grid.**

Then freeze and commit:

```bash
.venv/bin/python scripts/freeze_prereg.py
git add prereg scripts tests RUNBOOK_REVISION.md Makefile
git commit -m "Freeze two-tail pre-registration"
git log -1 --format=%h        # quote this hash in the paper next to 2bcfd96
```

Every later stage refuses to run until the lock matches. If you change the
pre-registration afterwards, `freeze_prereg.py --amend "reason"` keeps a history
entry that the paper must report.

## 2. The runs (tmux, resumable)

```bash
tmux new -s rev
bash scripts/run_revision.sh loud          # item 2.2 sweep + item 3.2 controls. Probes only
bash scripts/run_revision.sh loud-fv       # item 1.1: full-vocab basis, all 8 models, new folder
bash scripts/run_revision.sh quiet         # item 2.1: 11 concepts × 8 models × 2 layers + sweep
bash scripts/run_revision.sh quiet-length  # item 3.1: length-balanced and length-residualised
bash scripts/run_revision.sh loud-mid      # completes the 2 tails × 2 layers × 2 quantities family
bash scripts/run_revision.sh mlp           # item 3.3: non-linear probe, 4 models
```

Each stage writes to `results/_spectral/rev/<stage>/`, logs to
`results/_spectral/rev/logs/`, and skips anything already written. If it dies,
rerun the same command.

Runtime: multiply the smoke times. The loud stage costs about 6× one smoke run
per model (6 window sizes). The quiet stage is 88 model × concept cells × 2
layers × 6 window sizes. If that comes out too long, drop the 0.05 and 0.10
entries from the quiet sweep in `run_revision.sh` (not from the primary k),
and say so in the paper.

## 3. The report (item 5)

```bash
bash scripts/run_revision.sh report     # or: make paper
make fig1   # fig2 fig3 fig4 figS1 paper-numbers, one at a time
```

Output in `results/_spectral/rev/report/`:

| file | goes into the paper as |
|---|---|
| `fig1_loud_retention.pdf` | Figure 1: retention across the spectrum, loud tail |
| `fig2_quiet_retention.pdf` | Figure 2: retention across the spectrum, quiet tail |
| `fig3_per_model_delta.pdf` | Figure 3: per-model Δ with CIs, both tails |
| `fig4_placement_vs_capacity.pdf` | Figure 4: placement vs capacity |
| `figS1_k_sweep.pdf` | Appendix: Δ against window size |
| `table_two_tail.tex` | Table: per-model Δ, both tails |
| `paper_numbers.tex` | `\input` it. Every number in the text becomes a macro |
| `report.md` | every test in sentences; read this first |

`paper_numbers.tex` is the fix for the §4.2 problem. Numbers that come from a
macro cannot be copied from the wrong row.

## What each item on the list maps to

| item | where it's handled |
|---|---|
| 1.1 full-vocab p | `loud-fv` stage. The folder guard now refuses mixed bases, and `--audit` flags the old folder |
| 1.2 refusal numbers | text fix only; see `paper/revision_notes.md` |
| 1.3 gemma | every test in `report.md` is printed with and without gemma |
| 2.1 more concepts | `quiet` stage, 9 new concepts |
| 2.2 k sweep | `--k-fracs`, `k_sweep.csv`, `figS1` |
| 2.3 fix k first | `freeze_prereg.py`; stages refuse to run unfrozen |
| 2.4 multiplicity | one primary per tail plus a Holm family of 8, in `primary_and_family.csv` |
| 2.5 equivalence | TOST and one-sided non-superiority, same table |
| 3.1 length | `length.csv` (AUC per dataset) + `quiet-length` stage |
| 3.2 second loud control | `did_by_control`, `controls_loud.csv` |
| 3.3 non-linear probe | `--probe mlp`, `mlp` stage |
| 5.2 one command per figure | `make fig1` … `make figS1` |
