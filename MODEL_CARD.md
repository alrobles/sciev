# Model Card — Sciev v0.1.1

The `c_*` head weights are unchanged from v0.1. This patch release updates
the paper and audited metrics, fixes option-identity flip measurement,
and includes checksums and recorded dev temperatures.

> **Sciev** (`sci`·ence + `ev`·aluation): general-purpose typed
> decisions with a scientific specialty.

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
- **Heads**: one specialist per decision type, measured from the artifacts.
  - `c_choice`: **AttnPoolHead**, 67,166,209 parameters (268.7 MB) — a
    learned query pools each option span at layers {8,16,24,32}.
  - `c_noul`, `c_score`: 16,793,601 parameters (67.2 MB) each — MLP over
    final-layer mean-pooled spans; score adds an ordinal auxiliary loss.
- **Canonical ordering**: deterministic token-content sorting gives the
  same input sequence for distinct options under permutation. Duplicate or
  truncated-to-identical options require a separate external-ID tie policy.
- **Calibration**: temperatures fitted on elite dev, recorded to four
  decimals: choice **0.8274**, noul **2.7127**, score **0.9367**. Pass these
  explicitly as `--r2-temp`, or refit with `--r2-temp-fit-decisions`.

## Results

### Scientific decisions (elite battery, passage-grounded)

| type | n | acc | ECE | auto@5%err |
|---|---:|---:|---:|---:|
| choice | 653 | **0.8698** | 0.0253 | **0.8070** |
| noul | 1959 | 0.7825 | 0.0957 | 0.2777 |
| score | 1959 | 0.6075 | 0.1221 | 0.4334 |

### Public benchmarks (eval-only, zero-shot)

| benchmark | type | n | acc | ECE |
|---|---|---:|---:|---:|
| SciFact dev | noul | 340 | **0.8500** | 0.0998 |
| SciFact dev | score | 340 | 0.6294 | 0.0796 |
| SST-2 | choice | 872 | **0.9300** | 0.0682 |
| AG News | choice | 7600 | **0.8537** | 0.0556 |
| Enron spam | choice | 2000 | 0.7510 | 0.1046 |
| GPQA main | choice | 448 | 0.3125 | 0.3180 |
| GPQA diamond | choice | 198 | 0.3182 | 0.3089 |
| Banking77 (K=77) | choice | 3080 | 0.2289 | 0.0809 |

These are archived HPC results, not a new GPU evaluation during packaging.
`paper/results.json` includes source hashes, all metrics, and head-training
settings. Its values supersede earlier transcriptions in v0.1 documentation.
The canonical-order reports record flip=0; legacy non-canonical flip
figures used unaligned output positions and are withdrawn pending
re-evaluation. v0.1.1 compares original option identities instead.

## Intended use

- Decision layers in pipelines: routing, verification, grading,
  option selection — especially scientific/technical contexts
  (passage-grounded QA, claim verification against evidence).
- Confidence-based routing: `auto@5%err` is a retrospective diagnostic
  on labeled evaluation data. Validate thresholds on independent data
  before deployment; it is not a safety or error-rate guarantee.

## Out of scope / known limits

- **K > 4 options**: accuracy degrades (Banking77 0.23); the heads
  never saw more than 4 options. Chunked/tournament inference is future
  work.
- **Parametric knowledge QA** (GPQA): low accuracy under this evaluation
  protocol; these runs do not establish a theoretical backbone ceiling.
- **Score** is weaker than choice on the elite battery.
- Benchmarks may overlap backbone pretraining corpora (standard caveat).
- Results are single runs without multi-seed uncertainty estimates.
- External benchmarks were not used for gradient updates or temperature
  fitting, but did inform release-candidate selection.

## Training data

Scientific decision battery ("elite"): passage-grounded QA typed by
reasoning kind (factual, causal, multihop, negation, numerical,
definitional); passage-disjoint splits; groundedness filters
(content-word recall ≥ 0.5, numeric binding). No benchmark data in
training — all public-benchmark numbers are eval-only.

## Domain adaptation (DAPT)

The experimental LoRA candidates `dag_*` (g2000) and `da_*` (g5000)
did not satisfy the release gate. v0.1.1 retains frozen `c_*` and needs
no adapter. Table 4 reports the candidates and their limitations:

- Head learning rate, accumulation, warmup, and choice training duration
  differ from `c_*`; this is not a controlled DAPT-only ablation.
- g5000 warm-started from g2000 without optimizer/RNG restoration.
- 81.9M input token slots is a budget upper bound, not measured unique tokens.
- The 1.4B `bw1_sr` comparison is a 20-step checkpoint with shorter context,
  not a fully trained small-model baseline.

## Reproduce

```bash
python -m reverse_jev.eval --ckpt release/sciev-0.1-choice.pt \
    --r2-mode spanpool --r2-layers=-1,-9,-17,-25 --canonical-order \
    --r2-temp 0.8274 \
    --decisions-eval my_decisions.jsonl --device cuda --out eval.json
```

Benchmark converters: `data/convert_benchmarks.py` (GPQA, SciFact,
SST-2, AG News, Enron, Banking77). Battery builder:
`data/build_sci_decisions.py`. Full pipeline: `scripts/*.slurm`.

## Ecosystem position

A System-One-style decision layer on a **masked-diffusion** backbone,
with deterministic option presentation and a scientific training specialty.
This release makes no matched-protocol performance ranking against other
systems.

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
