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

Sciev uses a **masked-diffusion language model** (LLaDA-8B) and canonical
option sorting. Distinct option token sequences produce the same model
input under permutation; duplicate or truncated-to-identical options need
an explicit tie policy if their external identifiers must be distinguished.

## Results (v0.2, `systemone-v2` encoding, 3 seeds)

| eval | type | fr (release) mean ± sd | da (DAPT, experimental) |
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
release assets. A revised manuscript prepared for arXiv submission
reports this matched study with evidence controls and per-seed values.
Earlier v0.1.1 zero-shot numbers (SST-2 0.93, AG News 0.85)
remain archived in the release JSONs.

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
accuracy, weaker elite score than choice, and the option-side leakage
quantified by the evidence controls.

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
decision JSONL (the example assumes `my_decisions.jsonl` already exists):

```bash
pip install sciev
gh release download v0.2.2 --repo alrobles/sciev \
    --pattern 'fr_*.pt' --dir release
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

## Domain adaptation (DAPT)

`sciev/dapt.py` implements an experimental masked-token LoRA
adaptation loop for HF backbones (requires PEFT). Under the matched
comparison (same recipe, seeds and `systemone-v2` data), the adapted
backbone improves out-of-domain verification (SciFact noul +22.6pt) but
degrades in-domain ordinal scoring — the released heads therefore stay
frozen (`fr_*`), with `da_*` published as an experimental arm. See the
matched table and protocol caveats in the paper.

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

| path | qué |
|---|---|
| `sciev/model.py` | MdLMMoE + HFBackbone + heads + canonical layout |
| `sciev/train.py` | head training (CE, ordinal, RL-lite) |
| `sciev/eval.py` | acc/ECE/flip/auto@5%, temp-fit, remote API eval |
| `sciev/dapt.py` | domain-adaptive pretraining (LoRA) |
| `data/` | battery builders + benchmark converters |
| `scripts/*.slurm` | reproducible jobs (KU HPC) |
| `docs/PROJECT-STATUS.md` | roadmap, milestones, design decisions |
| `paper/main.tex` | archived release draft; the revised arXiv manuscript is prepared separately |

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
