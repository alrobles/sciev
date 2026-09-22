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

R2: two trained head variants over backbone hidden states — `marker`
(Laya-style, reads h at `[MASK]` slots before each option) and
`spanpool` (reads mean h over each observed option span). Train with
`reverse_jev.train --pairs-train`; evaluate with `eval --head/--r2-mode`.

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

## R2 — trained decision head (pairs_hard_v3 train, ~2k pairs, 6–8k steps)

Head `DecisionHead` over backbone hidden states; CE over the 2-option
softmax, order randomized per step. Both option orders evaluated per
pair (order-debiased acc + flip_rate = option-order sensitivity).

### marker mode (`[M]` before each option)

| ckpt | lvl | v3_eval | v4_holdout |
|---|---|---|---|
| esqueleto (freeze/ft) | all | ~0.50 | ~0.50 — head collapses to position noise (flip ~0.8) |
| f2-spanes-50k freeze | L2 | **0.730** | **0.751** |
| f2-spanes-50k ft | L2 | **0.777** | — |
| f2-spanes-50k ft | L3 | 0.516 | — |

The 10k-step backbone carries nothing readable at mask slots; the 50k
span-infilling backbone encodes stage grammar linearly at `[MASK]`
positions. Marker mode wins L2 outright.

### spanpool mode (mean h over each option's tokens)

| ckpt | lvl | v3_eval | v4_holdout |
|---|---|---|---|
| esq freeze | L0–L3 | 0.56–0.61 | 0.50–0.62 |
| f2 freeze | L2 | 0.692 | 0.676 |
| **f2 ft** | L0 | **0.757** | **0.729** |
| | L1 | **0.669** | **0.670** |
| | L2 | 0.712 | **0.706** |
| | L3 | 0.579 | 0.542 |
| | flip_rate | 0.29–0.43 | 0.35–0.43 (L3 ~0.89) |
| | brier | 0.175–0.24 | automation@5%: L0 0.22, L2 0.22 |

**f2-spanes-50k + spanpool + light FT is the best decision model so
far**: beats every zero-shot readout on every level except marker-ft on
L2. L3 holds ~0.54–0.58 across all readouts — the inferential ceiling
of this backbone; stable under debiased ordering but flip_rate ~0.89
means individual predictions remain order-fragile there.

Lesson so far: readout × backbone interact — token-space pseudo-
likelihood (r1_span), marker representations (L2 on f2), and pooled
span features each expose different signals. A Jev-scale model needs
the training objective to place the signal where the head reads it.

## Calibration (f2-spanes-50k + spanpool, v4_holdout_clean, n≈1900/lvl)

| model | lvl | acc | brier | ECE | NLL | auto@5% |
|---|---|---|---|---|---|---|
| CE (raw, T=1) | L0 | 0.729 | 0.179 | 0.026 | 0.544 | 0.086 |
| | L1 | 0.670 | 0.213 | 0.060 | 0.622 | 0.057 |
| | L2 | 0.706 | 0.182 | 0.060 | 0.539 | 0.161 |
| | L3 | 0.542 | 0.244 | 0.044 | 0.680 | 0.006 |
| CE+RL (raw) | L0 | **0.751** | 0.178 | 0.054 | 0.541 | 0.106 |
| | L1 | 0.664 | 0.214 | 0.044 | 0.621 | 0.064 |
| | L2 | 0.694 | 0.190 | 0.055 | 0.559 | 0.075 |
| | L3 | 0.544 | 0.244 | 0.046 | 0.681 | 0.013 |

Three findings:

1. **CE alone is nearly calibrated** — ECE 0.03–0.06 raw, matching the
   Laya/TypeSafe observation that supervised training captures most of
   the calibration; the reliability curve tracks tightly (L2: conf
   0.65→acc 0.70, 0.75→0.81, 0.85→0.91, 0.94→0.95).
2. **The scoring-rule RL term (RCDL-lite: REINFORCE over perturbed
   logits, log + 0.75·spherical reward) adds ~2 pts on L0** and keeps
   calibration honest — it refines distribution shape, not just argmax.
3. **Temperature transfer is domain-sensitive**: T=2.2–2.4 fitted on
   v3_eval barely helps (sometimes hurts) on v4_holdout — the batteries
   have different difficulty mixes, so one global T cannot fix
   cross-domain shift. Fit T on data matching deployment, or per-level.

L3 remains the frontier: ~0.54 holdout under every readout, confident
predictions concentrate near 0.5 (bin 0.52, n≈1600 → acc 0.54). The
model is honest about not knowing — which is itself a usable System-One
signal (route L3 to a slower reasoner).

## Tool-call decisions (ecological domain, eval = held-out lit sources)

`data/build_toolcall_decisions.py` converts the verified ecoreasoner
tool-call corpus into System-One decisions: `choice` (K=10 tools),
`noul` (is the proposed call valid?), `score` (0 wrong tool / 1 wrong
args / 2 correct). Train/dev from `toolcalls_fase3_500` (n=500); eval
from `lit_gold+pilot4+evolucion` (n=480, distinct sources — no leakage).
Battery: 2975 train / 525 dev / 3360 eval questions.

f2-spanes-50k + spanpool head, 6000 steps FT on all three kinds, T=1.18
fit on dev:

| kind | n | acc | chance | flip | brier | ECE | auto@5% |
|---|---|---|---|---|---|---|---|
| choice K=10 | 480 | 0.256 | 0.10 | 0.91 | 1.130 | 0.444 | 0.015 |
| noul K=2 | 1440 | 0.666 | 0.67 base | 0.51 | 0.449 | 0.040 | 0.0 |
| score K=3 | 1440 | 0.324 | 0.33 | 0.58 | 0.671 | 0.031 | 0.0 |

R1 zero-shot (same backbone, no training) is worse everywhere — choice
0.069 (below chance), noul 0.339, score 0.331, T=8.0 (≈uniform).

Honest read: real but weak signal on tool *choice* (2.5× chance, still
order-unstable); noul is near the majority-class rate and score is at
chance — the backbone lacks the semantic grounding to rank 10 ecological
tools or grade call correctness. Training was still climbing at 6k steps
(train_acc ~0.6); this is a floor, not a ceiling.

**Jev-1.13.0 full run** (all 3360 questions, 0 request errors):

| kind | n | Jev acc | ours | Jev brier | Jev ECE | Jev auto@5% |
|---|---|---|---|---|---|---|
| choice | 480 | **1.000** | 0.256 | 0.0005 | 0.005 | **1.00** |
| noul | 1440 | **0.887** | 0.666 | 0.168 | 0.060 | 0.78 |
| score | 1440 | **0.919** | 0.324 | 0.125 | 0.059 | 0.93 |
| overall | 3360 | **0.917** | ~0.45 | 0.126 | 0.051 | **0.90** |

Jev is not incrementally better — tool choice is *solved* (480/480,
near-deterministic calibrated probabilities) and 90% of all traffic is
automatable at a 5% error budget. The gap vs our 155M backbone is
capacity, not readout. This battery is the reference point: an open
System One that reaches Jev-level on eco tool routing is the bar.

```bash
python -m reverse_jev.eval --remote https://api.typesafe.ai \
    --api-key-file ~/env/typesafe-key --model-name jev-1.13.0 \
    --data toolcall_decisions_eval_text.jsonl --out jev_toolcall.json
```

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
