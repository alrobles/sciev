# Sciev

**Sciev** (`sci`·ence + `ev`·aluation) — open **System-One-style
decision models** on a frozen masked-diffusion backbone. Typed
questions (`choice` / `noul` / `score`) over a `state`, answered with
probabilities in **one forward pass**, calibrated with held-out dev data —
no text generation or output parsing. Classification errors remain possible.

*(repo: `alrobles/sciev`; `pip install sciev`)*

General-purpose by design, with a scientific specialty: the heads are
trained on passage-grounded scientific decisions and transfer zero-shot
to general classification.

## The System-One ecosystem

System One is now an open, multi-implementation ecosystem: Jev (closed),
OpenJev (third-party re-implementation on DiffusionGemma), and
Contrastive-LM's **CLM-8B** (Apache-2.0 weights, bi-encoder projections
over frozen Qwen3-8B). Sciev participates as the **masked-diffusion
member on a non-Qwen backbone**: it serves the same typed-question wire
contract (`/v1/systemone`, `/v1/rank`), so TypeSafe-style clients and
ecosystem harnesses can address it the same way they address CLM.

Where the implementations sit:

| | sciev | CLM-8B |
|---|---|---|
| encoder | LLaDA-8B, masked diffusion | Qwen3-8B, causal LM |
| state×option interaction | joint cross-encoding | bi-encoder dot product |
| decision heads | 3 typed heads, ~100M total | 2 projection heads, ~40M |
| training data | own conversions, hashed manifests, no proprietary-model outputs | Nemotron pairs + synthetic hard negatives + trajectories |
| specialty | evidence-grounded scientific decisions | agentic trajectories, program verification |
| audit trail | dataset/checkpoint fingerprints, disjoint-split protocol, fitted temperature + acceptance policy, declared known limits | paper-reported |

The backbone diversity is a feature of the ecosystem, not a rivalry:
CLM's heads are tied to the Qwen3-8B encoder, so environments whose
approved-model lists or provenance policies exclude Qwen-family weights
still have a wire-compatible option here. Their embedding-cache design
keeps the latency advantage on repeated states and unrestricted K;
sciev's release is bounded to ≤4 options per question (see known
limits). We also run the CLM architecture on our own backbone
(`sciev/biencoder.py`) as a matched-architecture control arm — same
data, same cohorts — so the joint-encoding-vs-dot-product comparison
inside the ecosystem is paired rather than confounded by encoder choice
or corpus scale.

Sciev uses a **masked-diffusion language model** (LLaDA-8B) and canonical
option sorting. Distinct option token sequences produce the same model
input under permutation; duplicate or truncated-to-identical options need
an explicit tie policy if their external identifiers must be distinguished.

## Results (v0.3, `systemone-v2` encoding)

v0.3.0 ships **anti-leakage heads**: `noul` and `score` were retrained on a
ρ=0.25 evidence-corruption mixture (for each clean row, 0.125 empty-evidence
and 0.125 shuffled-evidence counterparts labeled with the rejection class),
so corrupted evidence is rejected instead of pattern-matched on option-side
cues. `choice` is unchanged (`fr_choice.pt`) — an anti-leakage choice head
requires same-question confounders, which are covered in the manuscript but
not yet in the served weights.

| eval | type | v0.3 `alp25` | v0.2 `fr` (seed 7331) |
|---|---|---:|---:|
| sci battery elite (clean) | noul | 0.907 | 0.931 |
| sci battery elite (clean) | score | **0.911** | 0.896 |
| empty-evidence agreement | noul | **0.667** (prior) | 0.810 |
| shuffled-evidence agreement | noul | **0.667** (prior) | 0.669 |
| empty-evidence agreement | score | **0.336** (prior) | 0.575 |
| shuffled-evidence agreement | score | **0.336** (prior) | 0.385 |
| SciFact dev, canonical encoding | noul | **0.702** | 0.464 |
| SciFact dev | score | 0.757 | 0.757 |

Control rows report agreement with the original reference; the rejection
class is the correct behavior under corrupted evidence, so values at the
class prior (0.667 / 0.336) mark full rejection rather than failure.
Calibration: dev-fitted temperatures 1.403 (noul) and 0.7641 (score).

### v0.2 matched study (3 seeds, `fr` vs `da`)

| eval | type | fr (v0.2 heads) mean ± sd | da (DAPT, experimental) |
|---|---|---:|---:|
| sci battery elite | choice | **0.970 ± 0.017** | 0.961 ± 0.011 |
| sci battery elite | noul | **0.925 ± 0.006** | 0.913 ± 0.007 |
| sci battery elite | score | **0.859 ± 0.075** | 0.779 ± 0.109 |
| GPQA main | choice | 0.295 ± 0.014 | 0.288 ± 0.018 |
| GPQA diamond | choice | 0.307 ± 0.012 | 0.297 ± 0.011 |
| SciFact dev | noul | 0.483 ± 0.023 | **0.709 ± 0.020** |
| SciFact dev | choice | 0.479 ± 0.018 | 0.444 ± 0.062 |

Under matched seeds, recipe and data, DAPT helps out-of-domain
verification (SciFact noul +22.6pt) but degrades in-domain ordinal
scoring — task-dependent, not uniform. Evidence controls (empty /
shuffled passages) show `choice` retains option-side leakage: the
frozen head still agrees with the reference on **0.875** of items after
the passage is removed (adapted 0.806), while `score` is much more
sensitive to evidence changes; see the paper and `manifests/` in the
release assets. The [revised manuscript](https://sciev.org/static/sciev-paper.pdf?v=030),
in preparation for arXiv submission, reports this matched study with
evidence controls and per-seed values.
Earlier v0.1.1 zero-shot numbers (SST-2 0.93, AG News 0.85)
remain archived in the release JSONs.

<details><summary>Archived v0.1.x benchmark table (provenance)</summary>

| benchmark (archived v0.1.x, `c_*` heads) | type | n | acc | ECE | auto@5%err |
|---|---|---:|---:|---:|---:|
| sci battery (elite) | choice | 653 | **0.8698** | 0.0253 | **0.8070** |
| sci battery (elite) | noul | 1959 | 0.7825 | 0.0957 | 0.2777 |
| sci battery (elite) | score | 1959 | 0.6075 | 0.1221 | 0.4334 |
| SciFact dev | noul | 340 | **0.8500** | 0.0998 | 0.2706 |
| SciFact dev | score | 340 | 0.6294 | 0.0796 | 0.2559 |
| GPQA main | choice | 448 | 0.3125 | 0.3180 | 0.0112 |
| GPQA diamond | choice | 198 | 0.3182 | 0.3089 | 0.0152 |
| SST-2 (zero-shot) | choice | 872 | **0.9300** | 0.0682 | **0.9415** |
| AG News (zero-shot) | choice | 7600 | **0.8537** | 0.0556 | 0.3674 |
| Enron spam (zero-shot) | choice | 2000 | 0.7510 | 0.1046 | 0.0000 |
| Banking77 (K=77) | choice | 3080 | 0.2289 | 0.0809 | 0.0195 |

</details>

The archived v0.1.x table is kept for provenance; `paper/results.json`
records source hashes, metrics, and training settings. Caveat: the
elite-battery rows used a passage-first layout that truncated the
question tail on roughly half of long-context rows, so those cells
are retained as historical values, not clean question-conditioned
results — the `systemone-v2` study above supersedes them. Third-party
scores with unverified protocol equivalence are not included as direct
comparisons.

`auto@5%err` is the largest confidence-ranked prefix with realized error
≤ 5% on labeled evaluation data, not a deployment error guarantee.
v0.2 fits temperatures on dev data only, and calibration artifacts are
bound to the checkpoint SHA-256.

Known limits: weak K>4 performance (Banking77, v0.1.x), low GPQA
accuracy, and residual option-side leakage in `choice` (documented in
the manuscript; the v0.3 `alp25` heads removed it for `noul`/`score`).

## Architecture

```
ctx + question + options (canonically sorted by token-ids)
  -> frozen LLaDA-8B (bidirectional masked-diffusion LM)
  -> hidden states, layers {8,16,24,32}
  -> specialist head per decision type
       choice: AttnPoolHead — 67.2M parameters, four-layer span pooling
       noul:   MLP — 16.8M parameters, final-layer mean-pooled span
       score:  MLP — 16.8M parameters + ordinal auxiliary loss (P(y>=j))
  -> per-option logits -> softmax (temperature fitted on dev)
```

Checkpoint sizes are 269.4 MB (choice) and 68.6 MB each (noul, score),
measured from the serialized heads. The backbone stays frozen and is
fetched separately from HF. No DAPT adapter is required for the release.

## Quickstart

Install the package, download the release heads, and evaluate a prepared
decision JSONL (the example assumes `my_decisions.jsonl` already exists).
The same heads are also on Hugging Face at
[`alrobles/sciev`](https://huggingface.co/alrobles/sciev) (current release
heads under `release/` — `fr_choice.pt` plus `alp25_{noul,score}.pt`;
superseded v0.2 `fr_{noul,score}.pt` remain in place for provenance, and
experimental adapted heads are under `experimental/`).

```bash
pip install sciev
gh release download v0.3.0 --repo alrobles/sciev \
    --pattern '*.pt' --dir release
python -m sciev.eval \
    --ckpt release/fr_choice.pt --decision-type choice \
    --r2-mode spanpool --r2-layers=-1,-9,-17,-25 --canonical-order \
    --r2-temp-fit-decisions my_decisions_dev.jsonl \
    --decisions-eval my_decisions.jsonl \
    --device cuda --out eval.json
```

Temperatures are fitted on dev splits only (`--r2-temp-fit-decisions`);
the calibration artifact is bound to the checkpoint hash. Pass an
explicit `--r2-temp` to reuse a recorded value instead.

Decision records are `{ctx: [token-ids], opts: [[token-ids]...], gold}`
(`systemone-v2` also accepts text rows; see `sciev/decisions.py`).
Converters for GPQA / SciFact / SST-2 / AG News / Enron / Banking77 live
in `data/convert_benchmarks.py`; the scientific battery builder is
`data/build_sci_decisions.py`.

## Serving the wire contract

`sciev[serve]` exposes the TypeSafe-style HTTP API — `POST /v1/systemone`,
`POST /v1/rank`, `GET /v1/models`, `GET /health` — over a manifest that
binds backbone, head files and fitted temperatures (layout matches
`manifest.json` in the HF repo):

```bash
pip install "sciev[serve]"
SCIEV_MANIFEST=manifest.json uvicorn sciev.serve:app --port 8009

curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "The passage under decision.",
  "questions": {"q1": {"type": "noul",
    "instructions": "Does the passage support the claim?",
    "criteria": {"true": "supported", "false": "not supported"}}}
}'
```

Each answer carries calibrated `probabilities`, the `choice`/`noul`/
`score` decision and a `confidence` concentration score — one forward
pass per question. The server has no auth; keep it on localhost or
behind an authenticated proxy. A gated public deployment lives at
[sciev.ecoseek.org](https://sciev.ecoseek.org) (GitHub login).

## Domain adaptation (DAPT)

`sciev/dapt.py` implements an experimental masked-token LoRA
adaptation loop for HF backbones (requires PEFT). Under the matched
comparison (same recipe, seeds and `systemone-v2` data), the adapted
backbone improves out-of-domain verification (SciFact noul +22.6pt) but
degrades in-domain ordinal scoring — the release therefore stays on the
frozen backbone (`fr_choice` + `alp25_{noul,score}`), with `da_*`
published as an experimental arm. See the matched table and protocol
caveats in the paper.

```bash
python -m sciev.dapt \
    --hf-backbone GSAI-ML/LLaDA-8B-Instruct \
    --corpus papers.jsonl --field text \
    --out runs/dapt --steps 6000 --bs 16 --seq-len 1024
# then retrain heads with --lora-adapter runs/dapt/lora-final
```

`--resume` loads adapter weights and `--start-step` skips batches and
advances the schedule; it does **not** restore optimizer or RNG state.
It is a warm start, not an exact training-state resume. The DAPT module
accepts `--hf-backbone`, not `--ckpt`; native MdLMMoE checkpoints can be
used separately with `sciev.train --ckpt`.

## Repo map

| path | what |
|---|---|
| `sciev/model.py` | MdLMMoE + HFBackbone + heads + canonical layout |
| `sciev/train.py` | head training (CE, ordinal, RL-lite) |
| `sciev/eval.py` | acc/ECE/flip/auto@5%, temp-fit, remote API eval |
| `sciev/dapt.py` | domain-adaptive pretraining (LoRA) |
| `sciev/qa_train.py`, `sciev/qa_eval.py` | qa-v2 parametric-MCQA driver: budgeted training, disjoint temperature/policy fit |
| `sciev/protocol.py` | split disjointness, dataset/checkpoint contracts, fingerprints |
| `sciev/calibration.py` | dev-fitted temperature + acceptance-policy artifacts |
| `sciev/serve.py`, `sciev/inference.py` | `/v1/systemone` + `/v1/rank` wire API, manifest-bound engine |
| `sciev/biencoder.py` | CLM-architecture control arm (InfoNCE bi-encoder, matched data) |
| `data/` | battery builders + benchmark converters |
| `scripts/*.slurm` | reproducible jobs (KU HPC) |
| `docs/PROJECT-STATUS.md` | roadmap, milestones, design decisions |
| `paper/main.tex` | archived release draft; current manuscript served at [sciev.org](https://sciev.org/static/sciev-paper.pdf?v=030) |

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

The suite includes option-identity flip regressions and checks that paper
Tables 3 and 4 agree with `paper/results.json`. Compile the paper twice
with `pdflatex -interaction=nonstopmode -halt-on-error main.tex` from `paper/`.

## License

Apache-2.0 (code + heads). Backbones and datasets keep their own
licenses.
