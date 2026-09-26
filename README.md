# Sciev

**Sciev** (`sci`·ence + `ev`·aluation) — open **System-One-style
decision models** on a frozen masked-diffusion backbone. Typed
questions (`choice` / `noul` / `score`) over a `state`, answered with
probabilities in **one forward pass**, calibrated with held-out dev data —
no text generation or output parsing. Classification errors remain possible.

*(repo: `alrobles/sciev-devel`; package `sciev` for now)*

General-purpose by design, with a scientific specialty: the heads are
trained on passage-grounded scientific decisions and transfer zero-shot
to general classification.

Sciev uses a **masked-diffusion language model** (LLaDA-8B) and canonical
option sorting. Distinct option token sequences produce the same model
input under permutation; duplicate or truncated-to-identical options need
an explicit tie policy if their external identifiers must be distinguished.

## Results (v0.1.1, unchanged `c_*` weights from v0.1)

| benchmark | type | n | acc | ECE | auto@5%err |
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

Values come from the archived evaluation JSONs, not a new GPU evaluation.
`paper/results.json` records source hashes, metrics, and training settings.
These values supersede earlier transcriptions in the v0.1 documentation;
checkpoint hashes are unchanged. Third-party scores with unverified
protocol equivalence are not included as direct comparisons.

`auto@5%err` is the largest confidence-ranked prefix with realized error
≤ 5% on labeled evaluation data, not a deployment error guarantee.
The archived canonical-order reports record flip rate `0.00`. v0.1.1 fixes
the metric to compare original option identities, not permuted positions;
legacy non-canonical flip figures are withdrawn pending re-evaluation.

Known limits: weak K>4 performance (Banking77), low GPQA accuracy, and
weaker elite score than choice. Results are single runs. Public benchmarks
were not used for gradient updates or temperature fitting, but did inform
release-candidate selection.

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

Checkpoint sizes are 268.7 MB (choice) and 67.2 MB each (noul, score),
measured from the serialized heads. The backbone stays frozen and is
fetched separately from HF. No DAPT adapter is required for the release.

## Quickstart

From a v0.1.1 source checkout, download the heads and evaluate a prepared
decision JSONL (the example assumes `my_decisions.jsonl` already exists):

```bash
pip install -e .
gh release download v0.1.1 --repo alrobles/sciev-devel \
    --pattern 'sciev-0.1-*.pt' --dir release
python -m sciev.eval \
    --ckpt release/sciev-0.1-choice.pt \
    --r2-mode spanpool --r2-layers=-1,-9,-17,-25 --canonical-order \
    --r2-temp 0.8274 --decisions-eval my_decisions.jsonl \
    --device cuda --out eval.json
```

The `0.1` checkpoint filenames are intentional: v0.1.1 preserves those
weights. The recorded dev temperatures are choice **0.8274**, noul
**2.7127**, and score **0.9367**; pass the matching `--r2-temp` explicitly.
To refit from dev data instead, use `--r2-temp-fit-decisions`.

Decision records are `{ctx: [token-ids], opts: [[token-ids]...], gold}`.
Converters for GPQA / SciFact / SST-2 / AG News / Enron / Banking77 live
in `data/convert_benchmarks.py`; the scientific battery builder is
`data/build_sci_decisions.py`.

## Domain adaptation (DAPT)

`sciev/dapt.py` implements an experimental masked-token LoRA
adaptation loop for HF backbones (requires PEFT). The evaluated g2000 and
g5000 candidates did not satisfy the release gate, so v0.1.1 keeps `c_*`.
Their head optimization differed from `c_*`; the comparison does not
isolate DAPT as the cause of the observed transfer losses. See Table 4
and the protocol caveats in the paper.

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
| `paper/main.tex` | arXiv draft |

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
