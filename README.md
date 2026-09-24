# reverse-jev

Open **System-One-style decision models** on a frozen masked-diffusion
backbone. Typed questions (`choice` / `noul` / `score`) over a `state`,
answered with calibrated probabilities in **one forward pass** — no text
generation, nothing to parse, nothing to hallucinate.

General-purpose by design, with a scientific specialty: the heads are
trained on passage-grounded scientific decisions and transfer zero-shot
to general classification.

The only open entry in the System-One ecosystem
([kev](https://github.com/jaredpalmer/kev),
[openjev](https://huggingface.co/openjev/openjev),
[laya](https://huggingface.co/convaiinnovations/laya),
[SemIf](https://github.com/), djev-spark) built on a **masked-diffusion
language model** (LLaDA-8B) instead of an autoregressive one — and the
only one with **exact option-order invariance**: options are canonically
sorted by content, so the flip rate is `0.00` by construction rather
than ~0.03–0.08 empirically.

## Results (v0.1 candidate, `c_*` heads on frozen LLaDA-8B)

| benchmark | type | acc | ECE | auto@5%err | reference |
|---|---|---:|---:|---:|---|
| sci battery (elite) | choice | **0.870** | 0.025 | **0.807** | — |
| sci battery (elite) | noul | **0.783** | 0.096 | 0.278 | — |
| sci battery (elite) | score | **0.608** | 0.122 | 0.433 | — |
| SciFact dev | noul | **0.853** | 0.103 | 0.271 | ~0.89 |
| SciFact dev | score | 0.624 | 0.088 | 0.250 | ~0.70 |
| GPQA main / diamond | choice | 0.315 / 0.328 | 0.32 | 0.01 | backbone ceiling ~0.33 |
| SST-2 (zero-shot) | choice | **0.930** | 0.068 | **0.944** | 0.957 |
| AG News (zero-shot) | choice | **0.854** | 0.055 | 0.359 | 0.913 |
| Enron spam (zero-shot) | choice | 0.752 | 0.105 | 0.000 | 0.987 |
| Banking77 (K=77) | choice | 0.229 | 0.076 | 0.016 | 0.760 |

`auto@5%err` = share of decisions auto-acceptable while keeping realized
error ≤ 5% — the operating metric of a decision layer. Option-order flip
rate is `0.00` on every benchmark (canonical ordering).

Known limits: K>4 options (Banking77), parametric knowledge bounded by
the backbone (GPQA ≈ LLaDA ceiling), `score` is the weakest type for
every system measured.

## Architecture

```
ctx + question + options (canonically sorted by token-ids)
  -> frozen LLaDA-8B (bidirectional masked-diffusion LM)
  -> hidden states, layers {8,16,24,32}
  -> specialist head per decision type (~2M params)
       choice: AttnPoolHead — learned query pools each option's tokens
       noul:   MLP over mean-pooled span
       score:  MLP + CORAL-style ordinal auxiliary loss (P(y>=j))
  -> per-option logits -> softmax (temperature fitted on dev)
```

Heads are ~8 MB; the backbone stays frozen and is fetched from HF.
Only the heads ship with the release.

## Quickstart

```bash
pip install -e .
python -m reverse_jev.eval \
    --ckpt runs/sci/c_choice/decision.pt \
    --r2-mode spanpool --r2-layers=-1,-9,-17,-25 --canonical-order \
    --decisions-eval data/bench_external/gpqa/gpqa_main_choice_eval.jsonl \
    --device cuda --out eval.json
```

Decision records are `{ctx: [token-ids], opts: [[token-ids]...], gold}`.
Converters for GPQA / SciFact / SST-2 / AG News / Enron / Banking77 live
in `data/convert_benchmarks.py`; the scientific battery builder is
`data/build_sci_decisions.py`.

## Domain adaptation (DAPT)

`reverse_jev/dapt.py` continues the backbone's native masked-diffusion
objective (`mask ~ U(0,1)`, CE/t) on a domain corpus via LoRA — inject
domain knowledge into the weights, then retrain the heads on top:

```bash
python -m reverse_jev.dapt \
    --hf-backbone GSAI-ML/LLaDA-8B-Instruct \
    --corpus papers.jsonl --field text \
    --out runs/dapt --steps 6000 --bs 16 --seq-len 1024
# then retrain heads with --lora-adapter runs/dapt/lora-final
```

The same loop adapts any masked-diffusion checkpoint (e.g.
`MdLMMoE` from [alrobles/ecoreasoner](https://github.com/alrobles/ecoreasoner)
via `--ckpt`).

## Repo map

| path | qué |
|---|---|
| `reverse_jev/model.py` | MdLMMoE + HFBackbone + heads + canonical layout |
| `reverse_jev/train.py` | head training (CE, ordinal, RL-lite) |
| `reverse_jev/eval.py` | acc/ECE/flip/auto@5%, temp-fit, remote API eval |
| `reverse_jev/dapt.py` | domain-adaptive pretraining (LoRA) |
| `data/` | battery builders + benchmark converters |
| `scripts/*.slurm` | reproducible jobs (KU HPC) |
| `docs/PROJECT-STATUS.md` | roadmap, milestones, design decisions |
| `paper/main.tex` | arXiv draft |

## License

Apache-2.0 (code + heads). Backbones and datasets keep their own
licenses. Third-party published numbers are cited, not reproduced.
