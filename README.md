# reverse-jev

Open [System One](https://docs.typesafe.ai/concepts/system-one)-style decision
models on the EcoReasoner dLLM backbone. Typed questions (`noul` / `choice` /
`score`) over a `state`, answered with calibrated probabilities in **one
forward pass** — no text generation, nothing to parse, nothing to hallucinate.

A deconstruction-and-rebuild of TypeSafe AI's **Jev** (released 2026-09-15,
API-only), in the spirit of [Laya](https://huggingface.co/convaiinnovations/laya)
and [Kev](https://github.com/jaredpalmer/kev), but on our own masked-diffusion
backbone (`MdLMMoE`, vendored from
[alrobles/ecoreasoner](https://github.com/alrobles/ecoreasoner)) and aimed at
scientific/ecological decisions. Design rationale:
`docs/designs/ecoreasoner-Fase4-SYSTEMONE-DESIGN.md` in ecoreasoner.

## Status

Early scaffold. The backbone checkpoints live on KU HPC
(`/beegfs/a474r867/ecoreasoner/runs/`); everything here runs CPU-friendly for
development.

## The readout (R1, zero extra params)

```
state + question + [MASK]  ->  logits[MASK] sliced to each option's
                               first-token id set  ->  softmax  ->  P(option)
```

A single mask slot after the prompt; each option maps to its candidate
first-token ids (casing/space variants). `noul` slices over {yes,true,si}
vs {no,false}; `choice` over option names; `score` over level-index tokens.
Confidence is a statistic of the distribution: `(p_max − 1/K)/(1 − 1/K)`.
Temperature scales logits post-hoc (fitted on held-out dev data only).

R2 (Laya-style marker head over full-text options) is stubbed in
`model.DecisionHead` — the next step once E0 gives signal.

## E0 — the no-training experiment

Does a dedicated decision readout discriminate better than the Fase-3
denoise-loss scorer on the same pairs? Three readouts compared:

- `r1_first_token`: `ctx + [MASK]`, compare logits of each candidate's
  first token. Diagnostic only — on these batteries `first_token_differs`
  is 0 (pairs diverge deeper in the span), so this is degenerate.
- `r1_span`: `ctx + [MASK]*len(cand)`, mean logprob of true candidate
  tokens at masked positions — the dLLM-native option scorer.
- `legacy_denoise`: random-mask denoise CE over `ctx + cand` (Fase-3).

```bash
python -m reverse_jev.eval \
    --ckpt /beegfs/a474r867/ecoreasoner/runs/f0-span-esqueleto/checkpoint-g10000/model.pt \
    --config harness/configs/f0-span-esqueleto.yaml \
    --pairs /beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3/pairs_L3.jsonl
```

### E0 result — pairwise_acc, `r1_span` vs `legacy_denoise`

pairs_hard_v3_eval (n≈500/level) / pairs_hard_v4_holdout_clean (n≈1900/level):

| ckpt | lvl | v3 legacy | v3 span | v4 legacy | v4 span |
|---|---|---|---|---|---|
| f0-span-esqueleto | L0 | 0.540 | **0.600** | 0.555 | **0.606** |
| | L1 | 0.537 | **0.593** | 0.530 | **0.564** |
| | L2 | 0.611 | 0.613 | 0.573 | **0.612** |
| | **L3** | 0.518 | **0.579** | 0.527 | **0.571** |
| f0-span-v2-weight-tying | L0 | 0.511 | 0.544 | 0.526 | 0.521 |
| | L1 | 0.506 | 0.522 | 0.512 | **0.543** |
| | L2 | 0.617 | **0.639** | 0.613 | **0.657** |
| | **L3** | 0.516 | **0.568** | 0.511 | **0.577** |
| f0-span-esqueleto-v2-piloto | L0 | 0.501 | 0.535 | 0.524 | 0.525 |
| | L1 | 0.514 | 0.526 | 0.509 | **0.539** |
| | L2 | 0.604 | **0.617** | 0.593 | **0.632** |
| | **L3** | 0.524 | **0.554** | 0.511 | **0.564** |
| f2-spanes-50k | L0 | 0.568 | **0.716** | 0.572 | **0.741** |
| | L1 | 0.555 | **0.658** | 0.553 | **0.664** |
| | L2 | 0.611 | 0.525 | 0.586 | 0.524 |
| | **L3** | 0.512 | **0.575** | 0.522 | **0.560** |

**The readout was the bottleneck.** All four checkpoints — L3 ≈ 0.51–0.53
(chance) under denoise-loss, the result that falsified the inferential
thesis at the 0.55 gate — land at L3 ≈ 0.56–0.58 on both batteries when
the candidate span is scored under a full mask (~4σ over chance at
n=1767). f2-spanes-50k additionally reveals strong shallow-discourse
signal (L0 0.74, L1 0.66 holdout) that denoise-loss hid; its L2
inversion suggests stage-grammar and option-content trade off under
different readouts — worth its own probe.

Caveats: pairwise discrimination ≠ calibration (RLCD/temperature come
later); `r1_span` is mean logprob — length-normalized, not a true joint;
the same tokenizer/model pair must score both candidates.

## Evaluate the real Jev (or any System One endpoint)

```bash
python -m reverse_jev.eval --remote https://api.typesafe.ai \
    --api-key-file ~/env/typesafe-key --data evals/eco_decisions.jsonl
```

Reports accuracy / Brier / ECE / automation@5%-error per question type —
the same numbers our local model reports, so comparisons are apples-to-apples.

## Convert ecoreasoner pairs to decision data

```bash
python data/convert_pairs.py --pairs pairs_L3.jsonl \
    --tokenizer GSAI-ML/LLaDA-8B-Instruct --out evals/pairs_L3_decisions.jsonl
```

## Train (RCDL-lite)

```bash
python -m reverse_jev.train --data train.jsonl --dev dev.jsonl \
    --ckpt runs/f0/checkpoint-g10000/model.pt --out runs/dec-001 \
    --steps 2000 --lr 2e-5 --accum 8            # CE only
# + --rl 1.0 for the REINFORCE scoring-rule term (log + 0.75·spherical)
```

Post-hoc temperature is fitted on `--dev` and saved to `temperature.json`.

## Serve (TypeSafe-SDK-compatible)

```bash
REVJEV_CKPT=runs/dec-001/model.pt uvicorn reverse_jev.serve:app --port 8009
```

```python
from typesafe_sdk import TypeSafeClient, Noul
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009")
client.system_one(state="...", questions={"q": Noul(instructions="...")})
```

## Layout

| path | what |
|---|---|
| `reverse_jev/model.py` | MdLMMoE (vendored v2: RoPE/weight-tying/MoE) + DecisionHead + checkpoint loader |
| `reverse_jev/readout.py` | R1 token-slot readout: options → id sets → sliced softmax |
| `reverse_jev/eval.py` | pairs E0 (R1 vs denoise_loss) + decision metrics (acc/Brier/ECE/automation) + remote Jev eval |
| `reverse_jev/train.py` | decision fine-tune: sliced-softmax CE + optional REINFORCE scoring rule + temperature fit |
| `reverse_jev/serve.py` | `POST /v1/systemone` compatible with `typesafe-sdk` |
| `reverse_jev/data.py` | pairs/System-One JSONL loaders, dev/test split |
| `data/convert_pairs.py` | ecoreasoner pairs → decision JSONL |
| `tests/test_smoke.py` | CPU smoke tests, tiny random model |

## License

MIT — A.L. Robles Fernández
