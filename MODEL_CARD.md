# Model Card — Sciev v0.2.2

The `fr_*` frozen head weights are unchanged from the v0.2 weight set;
v0.2.2 is a packaging refresh that ships manifests, checksums and
audited metrics. The `da_*` adapted heads are published as an
experimental arm and are not the released recipe.

> **Sciev** (`sci`·ence + `ev`·aluation): open System-One scientific
> decision models — typed decisions over supplied evidence, with a
> scientific specialty.

## What it is

A **typed decision model**: unstructured `state` + typed questions in →
structured decisions with probabilities out, in **one forward pass per
decision**. No text generation, no output parsing, no format
hallucination. Classification errors remain possible.

Three decision types:

| type | semantics | K |
|---|---|---|
| `choice` | pick one of K labelled options | 2–4 trained |
| `noul` | is the claim supported by the state? | 2 |
| `score` | ordinal grade on a fixed rubric | 3 (0<1<2) |

## Architecture

- **Backbone**: `GSAI-ML/LLaDA-8B-Instruct` — masked-diffusion LM,
  frozen. Bidirectional attention by construction (no causal mask), so
  option spans see the full context symmetrically. Inference reads a
  fully observed sequence; it does not run a denoising loop.
- **Heads**: one specialist per decision type, measured from the artifacts.
  - `fr_choice`: **AttnPoolHead**, 67,166,209 parameters (269.4 MB) — a
    learned query pools each option span at hidden-state outputs
    (-1,-9,-17,-25) ≡ layers {32,24,16,8}.
  - `fr_noul`, `fr_score`: 16,793,601 parameters (68.6 MB) each — MLP over
    final-layer mean-pooled spans; score adds an ordinal auxiliary loss.
- **Canonical ordering**: deterministic token-content sorting gives the
  same input sequence for distinct options under permutation. Duplicate or
  truncated-to-identical options are rejected, not silently disambiguated.
- **Calibration**: a positive scalar temperature is fitted per checkpoint
  on the corresponding `systemone-v2` dev split; the artifact is bound to
  the checkpoint SHA-256. Refit with `--r2-temp-fit-decisions` or pass
  `--r2-temp`. Temperature changes probabilities, not the argmax.

## Results (matched v0.2 study, `systemone-v2`)

Mean ± sample SD across three head-training seeds (7331–7333); same head
recipe for both arms. `da` uses one experimentally adapted backbone —
see *Domain adaptation* for why this is not a causal DAPT estimate.

| eval | type | n | fr (release) | da (experimental) |
|---|---|---:|---:|---:|
| sci battery | choice | 448 | **0.970 ± 0.017** | 0.961 ± 0.011 |
| sci battery | noul | 1200 | **0.925 ± 0.006** | 0.913 ± 0.007 |
| sci battery | score | 1218 | **0.859 ± 0.075** | 0.779 ± 0.109 |
| GPQA main | choice | 441 | 0.295 ± 0.014 | 0.288 ± 0.018 |
| GPQA diamond | choice | 193 | 0.307 ± 0.012 | 0.297 ± 0.011 |
| SciFact dev | noul | 332 | 0.483 ± 0.023 | **0.709 ± 0.020** |
| SciFact dev | choice (3-way) | 332 | 0.479 ± 0.018 | 0.444 ± 0.062 |

External evaluations informed candidate selection; treat them as
exploratory, selection-informed estimates. GPQA main and diamond overlap.

### Evidence controls (seed 7331)

Controls keep the original reference label but alter the evidence, so
their metric is *agreement with the original reference*, not accuracy:

| task | arm | intact acc | empty agree | shuffled agree |
|---|---|---:|---:|---:|
| choice | fr | 0.9799 | 0.8750 | 0.8725 |
| choice | da | 0.9598 | 0.8058 | 0.7852 |
| noul | fr | 0.9308 | 0.8100 | 0.6693 |
| noul | da | 0.9200 | 0.8067 | 0.6667 |
| score | fr | 0.8957 | 0.5747 | 0.3848 |
| score | da | 0.7521 | 0.5550 | 0.4300 |

The internal battery's high accuracy does **not** establish dependence on
the supplied passage: choice retains 87.5% reference agreement with the
passage removed. Score is more evidence-sensitive, but sensitivity is
not proof of semantic grounding.

### Archived v0.1.x results (`c_*` heads)

| benchmark | type | n | acc | ECE | auto@5%err |
|---|---|---:|---:|---:|---:|
| sci battery (elite) | choice | 653 | 0.8698 | 0.0253 | 0.8070 |
| sci battery (elite) | noul | 1959 | 0.7825 | 0.0957 | 0.2777 |
| sci battery (elite) | score | 1959 | 0.6075 | 0.1221 | 0.4334 |
| SciFact dev | noul | 340 | 0.8500 | 0.0998 | 0.2706 |
| SciFact dev | score | 340 | 0.6294 | 0.0796 | 0.2559 |
| GPQA main | choice | 448 | 0.3125 | 0.3180 | 0.0112 |
| GPQA diamond | choice | 198 | 0.3182 | 0.3089 | 0.0152 |
| SST-2 (zero-shot) | choice | 872 | 0.9300 | 0.0682 | 0.9415 |
| AG News (zero-shot) | choice | 7600 | 0.8537 | 0.0556 | 0.3674 |
| Enron spam (zero-shot) | choice | 2000 | 0.7510 | 0.1046 | 0.0000 |
| Banking77 (K=77) | choice | 3080 | 0.2289 | 0.0809 | 0.0195 |

> **Known defect in the elite battery**: its passage-first layout
> truncated the question tail on roughly half of long-context rows, so
> the elite cells are historical values, not clean question-conditioned
> results. The `systemone-v2` matched study supersedes them. External
> benchmark rows (GPQA, SciFact, SST-2…) are unaffected by this defect.
> `auto@5%err` is a retrospective oracle diagnostic on labeled
> evaluation data, not a deployment error guarantee.

## Intended use

- Decision layers in pipelines: routing, verification, grading,
  option selection — especially scientific/technical contexts
  (passage-grounded QA, claim verification against evidence).
- Confidence-based routing: validate any acceptance threshold on
  independent labeled data for your distribution before deployment;
  dev-fitted calibration does not transfer automatically across tasks.

## Out of scope / known limits

- **Evidence dependence**: the internal battery's references are partly
  heuristic (constructed distractors, synthetic score labels); use the
  evidence controls to judge task validity, not accuracy alone.
- **K > 4 options**: accuracy degrades (Banking77 0.23); the heads
  never saw more than 4 options. Chunked/tournament inference is future
  work.
- **Parametric knowledge QA** (GPQA): low accuracy under this protocol;
  these runs do not establish a backbone knowledge ceiling.
- Benchmarks may overlap backbone pretraining corpora (standard caveat).
- The v0.2 matched study replicates three head seeds on **one** adapted
  backbone; it does not measure adaptation-seed variance.
- External benchmarks were not used for gradient updates or temperature
  fitting, but did inform release-candidate selection.

## Training data

Scientific decision battery (`systemone-v2`): passage-grounded QA typed
by reasoning kind; lexical consistency filters (content-word recall
≥ 0.5, numeric binding); source/passage-grouped splits; complete-input
encoding (long rows excluded rather than truncated). No benchmark data
in training or temperature fitting — public-benchmark numbers are
eval-only. The older `elite-v1` set is retired from headline claims due
to the truncation defect above.

## Domain adaptation (DAPT)

`da_*` heads pair the `scientific-v1` recipe with one archival LoRA
adapter (rank 16, 5,000 nominal steps). Under the matched comparison the
adapted arm improves out-of-domain verification (SciFact noul +22.6pt
mean) but degrades in-domain ordinal scoring — task-dependent, not a
uniform effect. Caveats on the historical adapter run:

- Its loss divides mean masked-token CE by the batch-mean mask rate; it
  is **not** the native per-token importance-weighted LLaDA estimator.
- The final segment warm-started at step 2000 without optimizer/RNG
  restoration; 81.9M token slots is a budget bound, not unique tokens.
- One adapted backbone replicate; results do not generalize to a recipe.

## Reproduce

```bash
python -m sciev.eval --ckpt release/fr_choice.pt \
    --decision-type choice \
    --r2-mode spanpool --r2-layers=-1,-9,-17,-25 --canonical-order \
    --r2-temp-fit-decisions my_decisions_dev.jsonl \
    --decisions-eval my_decisions.jsonl --device cuda --out eval.json
```

Benchmark converters: `data/convert_benchmarks.py` (GPQA, SciFact,
SST-2, AG News, Enron, Banking77). Battery builder:
`data/build_sci_decisions.py`. Full pipeline: `scripts/*.slurm`.

## Ecosystem position

An open System-One-style decision layer on a **masked-diffusion**
backbone, with deterministic option presentation and a scientific
training specialty. Other open systems (e.g., OpenJev on DiffusionGemma)
also use diffusion backbones; this release makes no matched-protocol
performance ranking against other systems.

## License

Apache-2.0 (code + heads). Backbone: LLaDA license. Datasets: own
licenses. No proprietary model outputs used in training.

## Citation

```
@misc{robles_fernandez_sciev_2026,
  title  = {Sciev: Typed Scientific Decisions Without Generation},
  author = {Robles-Fern\'andez, Angel Luis},
  year   = {2026},
  note   = {arXiv preprint (in preparation)}
}
```
