# Design provenance

This repo implements the line proposed in
`alrobles/ecoreasoner/docs/designs/ecoreasoner-Fase4-SYSTEMONE-DESIGN.md`
(2026-09-21): reformulate the falsified "dLLM generates the inferentially
correct continuation" thesis as "dLLM scores typed options with calibrated
probabilities" — the System One contract that TypeSafe's Jev popularized.

ecoreasoner keeps the corpus, the L0–L3 pair batteries, and the research log.
sciev keeps the decision readout, the RLCD-lite trainer, the
SDK-compatible server, and the calibration harness that compares our model
against the real Jev API on identical questions.

Reference implementations studied:
- Laya (convaiinnovations/laya): ModernBERT-large + option-marker scorer head,
  REINFORCE over Gaussian-perturbed distributions, reward = log + 0.75·spherical.
- Kev (jaredpalmer/kev): LoRA + pointer head on Qwen3.5, `/v1/systemone`
  compatible, temperature ~2.1–2.4 fitted on dev.
- OpenJev (razorback16/openjev): DiffusionGemma denoising of answer slots.

## Alignment implementation plan (after v0.1.1)

Research question: can an open typed decision layer make evidence-grounded
scientific selections, verification judgments, and rubric-based ratings,
with useful calibration and measured selective risk? API compatibility is
an interface requirement, not evidence of scientific validity. Generative
reasoning, parametric knowledge, and evidence-grounded decisions must be
evaluated separately.

### Priorities and acceptance criteria

| Priority | Workstream | Acceptance |
|---|---|---|
| P0 | Data isolation and provenance | Negative pools are split-local; numerical normalization preserves values; malformed rows fail explicitly; source/group/type/encoding metadata survive loading. |
| P0 | Shared R2 input contract | Training, temperature fitting, evaluation and API use one layout implementation; option identity survives permutations; token budgets and vocabulary are validated. |
| P1 | Comparable training | Frozen/adapted comparisons use the same per-type recipe; existing incompatible output directories are not silently reused; checkpoint inference settings and data fingerprints are recorded. |
| P1 | Calibration and evaluation | Fit temperature and acceptance policy on dev only, then evaluate a frozen policy on test; tied confidences cannot be split to cherry-pick a prefix; nominal and ordinal tasks have distinct metrics. |
| P1 | R2 serving | One shared backbone, real per-type heads, explicit temperatures, full criteria/instructions, deterministic JSON encoding, and CPU end-to-end tests without model downloads. |
| P2 | Scientific controls and verification | No-evidence and altered-evidence controls, consistent sample identity, full integration tests, and an explicit protocol for later external comparisons. |

Historical release artifacts and `paper/results.json` remain immutable.
Correcting a pipeline does not establish new accuracy or validate old
calibration on a changed encoding. The reference heads can be inspected
and loaded, but changed text protocols require new validation before their
old temperature/automation claims are reused.

### Shared implementation interfaces

`sciev.decisions` owns the framework-independent data contract and
R2 preprocessing:

- `ENCODING_VERSION = "systemone-v2"`.
- `validate_decision_row(row, require_gold=True)` returns a validated copy
  preserving metadata; invalid labels, token types, empty options, and
  invalid soft distributions raise `ValueError`.
- `encode_question(tokenizer, state, question, max_ctx=640, max_opt=120,
  overflow="error")` returns `ctx`, `opts`, `kind`, `option_keys`,
  `encoding`, and `schema_version`. It never uses question IDs or labels as
  features. Structured state/instructions/criteria use deterministic JSON.
  Choice descriptions and optional noul criteria are part of the input;
  score instructions are not silently replaced with a noul question.
- `prepare_decision(model, ctx, opts, mode="spanpool", canonical=False,
  order=None, strict=False)` returns a `PreparedDecision` with `ids`,
  `positions`, `order` (presented index to original index),
  `context_truncated`, and `option_truncated`. It validates vocabulary and
  rejects indistinguishable option encodings. `strict=True` rejects loss
  of input tokens; the non-strict path reports truncation explicitly.
- `decision_logits(model, head, ctx, opts, device, mode="spanpool",
  layers=(-1,), canonical=False, order=None, strict=False)` returns
  `(logits_in_original_option_order, prepared)` using one backbone forward.

`sciev.metrics` owns validated distribution metrics and dev-fitted
acceptance policies. Per-class/majority metrics require a fixed semantic
label space; option positions in arbitrary QA are not semantic classes.
Acceptance policies use maximum class probability, not the API's
concentration-style confidence field, and carry no unearned deployment
risk guarantee.

### Parallel ownership

- Data agent: dataset builders/converters, `sciev/data.py`, and their tests.
- Metrics agent: `sciev/metrics.py` and its tests.
- API agent: `sciev/inference.py`, `sciev/serve.py`, and their tests.
- Coordinator: shared decisions, training/evaluation integration, experiment
  scripts, documentation, and cross-component verification.

No paid API calls, model downloads, HPC submissions, commits, pushes or
release changes are delegated to subagents. Integration is performed on
`alignment/scientific-pipeline`, with failing regression tests preceding fixes.

### Implementation status

Implemented on this branch (software contracts only; no new accuracy claims):

- `sciev.protocol`: matched `scientific-v1` recipe per decision type,
  dataset identity contracts (exact input, declared source/document/group
  provenance), disjointness assertions between calibration/evaluation/training
  roles, checkpoint-vs-data overlap checks (`unverified` when a legacy
  checkpoint lacks recorded training identities), and fresh-output guards.
- `sciev.calibration`: dev-only temperature and acceptance-policy
  fitting through the shared decision path, frozen JSON artifacts bound to
  the checkpoint SHA-256, load-time validation of evaluation/calibration
  disjointness and expected inference settings. Legacy checkpoints report
  training provenance as `unverified` rather than passing silently.
- `sciev.train`/`eval`: `--recipe scientific-v1`, per-row strict-input
  handling for `systemone-v2`, self-describing checkpoints (`inference`
  settings, `training_data` contract, input file fingerprints), final
  partial-accumulation optimizer step, dev/eval identity-overlap checks,
  `--calibration-in`, `--train-reference`, `--decision-type`, `--fixed-labels`.
- `sciev.inference`/`serve`: manifest-based R2 engine over the shared
  encoding; explicit legacy opt-in for historical bundles; API/local parity
  including canonical tie selection and maximum-probability policies.
- `sciev.data` and builders: split-local negative pools, connected
  group splitting, NFC-preserving content identity (case and compatibility
  symbols are scientifically significant), Decimal numeric handling,
  explicit evidence spans, and `make_evidence_controls` producing
  reference-labelled (not relabelled) no-evidence/shuffled-evidence controls.
- `scripts/sci_llada6.slurm`: paired frozen/adapted arms under fresh
  `da_matched_s$SEED` tags on `systemone-v2` data; refuses existing outputs.

Still required before any new result claims: regenerate `systemone-v2`
data, run matched training/calibration/evaluation on GPU, evaluate the
frozen acceptance policy on untouched test data, run evidence controls,
and document results separately from archived v0.1.1 numbers.
