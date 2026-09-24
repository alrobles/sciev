# Model Card — Sciev v0.1

> Working name: **Sciev** (`sci`·ence + `ev`·aluation) — final branding TBD.

## What it is

A **typed decision model**: unstructured `state` + typed questions in →
structured decisions with calibrated probabilities out, in **one forward
pass**. No text generation, no output parsing, no format hallucination.

Three decision types:

| type | semantics | K |
|---|---|---|
| `choice` | pick one of K labelled options | 2–4 trained |
| `noul` | is the claim supported by the state? | 2 |
| `score` | ordinal grade on a fixed rubric | 3 (0<1<2) |

## Architecture

- **Backbone**: `GSAI-ML/LLaDA-8B-Instruct` — masked-diffusion LM,
  frozen. Bidirectional attention by construction (no causal mask), so
  option spans see the full context symmetrically.
- **Heads** (~2M params each, ~8MB): one specialist per decision type.
  - `c_choice`: **AttnPoolHead** — learned query attends over each
    option's token span; features concatenated from layers {8,16,24,32}.
  - `c_noul`, `c_score`: MLP over mean-pooled spans; score adds a
    CORAL-style ordinal auxiliary loss.
- **Canonical ordering**: options sorted deterministically by token-id
  content before encoding → permutation invariance is *exact*, not
  empirical (flip rate 0.00 by construction on every benchmark).
- **Calibration**: per-type temperature fitted on held-out dev.

## Results

### Scientific decisions (elite battery, passage-grounded)

| type | acc | ECE | auto@5%err | flip |
|---|---:|---:|---:|---:|
| choice | **0.870** | 0.025 | **0.807** | 0.00 |
| noul | 0.783 | 0.096 | 0.278 | 0.00 |
| score | 0.608 | 0.122 | 0.433 | 0.00 |

### Public benchmarks (eval-only, zero-shot)

| benchmark | type | acc | ECE | reference |
|---|---|---:|---:|---|
| SciFact dev | noul | **0.853** | 0.103 | ~0.89 (open SOTA) |
| SciFact dev | score | 0.624 | 0.088 | ~0.70 |
| SST-2 | choice | **0.930** | 0.068 | 0.957 |
| AG News | choice | **0.854** | 0.055 | 0.913 |
| Enron spam | choice | 0.752 | 0.105 | 0.987 |
| GPQA main | choice | 0.315 | 0.319 | backbone ceiling ~0.33 |
| Banking77 (K=77) | choice | 0.229 | 0.076 | 0.760 |

Reference column: third-party published measurements on identical
datasets (Kumar 2026; open-ecosystem reports), not our own runs.

## Intended use

- Decision layers in pipelines: routing, verification, grading,
  option selection — especially scientific/technical contexts
  (passage-grounded QA, claim verification against evidence).
- Automation under a budget: `auto@5%err` quantifies the share of
  traffic safely auto-acceptable; low-margin cases are meant to be
  escalated, not guessed.

## Out of scope / known limits

- **K > 4 options**: accuracy degrades (Banking77 0.23); the heads
  never saw more than 4 options. Chunked/tournament inference is future
  work.
- **Parametric knowledge QA** (GPQA): bounded by the frozen backbone's
  ceiling (LLaDA-8B ≈ 0.33). The readout extracts what's accessible;
  it does not add knowledge. Domain knowledge belongs in the backbone
  (see DAPT below).
- **Score** is the weakest type for every measured system including the
  proprietary reference.
- Benchmarks may overlap backbone pretraining corpora (standard caveat).

## Training data

Scientific decision battery ("elite"): passage-grounded QA typed by
reasoning kind (factual, causal, multihop, negation, numerical,
definitional); passage-disjoint splits; groundedness filters
(content-word recall ≥ 0.5, numeric binding). No benchmark data in
training — all public-benchmark numbers are eval-only.

## Domain adaptation (DAPT)

`reverse_jev.dapt`: continue the backbone's masked-diffusion objective
on a domain corpus via LoRA → `--lora-adapter` at head training.
The adapter path is baked into `decision.pt` and merged at load time.

## Reproduce

```bash
python -m reverse_jev.eval --ckpt runs/sci/c_choice/decision.pt \
    --r2-mode spanpool --r2-layers=-1,-9,-17,-25 --canonical-order \
    --r2-temp-fit-decisions data/sci_battery/sci_choice_dev.jsonl \
    --decisions-eval <benchmark>.jsonl --device cuda --out eval.json
```

Benchmark converters: `data/convert_benchmarks.py` (GPQA, SciFact,
SST-2, AG News, Enron, Banking77). Battery builder:
`data/build_sci_decisions.py`. Full pipeline: `scripts/*.slurm`.

## Ecosystem position

The only open System-One-style entry on a **masked-diffusion** backbone
(kev/openjev/laya are autoregressive or encoder-based) and the only one
with **exact** option-order invariance (0.00 by construction vs.
~0.03–0.08 empirical elsewhere). General-purpose by design; scientific
specialty by training data.

## License

Apache-2.0 (code + heads). Backbone: LLaDA license. Datasets: own
licenses. No proprietary model outputs used in training.

## Citation

```
@misc{sciev2026,
  title  = {Open System-One Decision Models: Frozen Diffusion-LM
            Backbones and Specialist Decision Heads},
  author = {Robles, A.},
  year   = {2026},
  note   = {arXiv preprint (in preparation)}
}
```
