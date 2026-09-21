# Design provenance

This repo implements the line proposed in
`alrobles/ecoreasoner/docs/designs/ecoreasoner-Fase4-SYSTEMONE-DESIGN.md`
(2026-09-21): reformulate the falsified "dLLM generates the inferentially
correct continuation" thesis as "dLLM scores typed options with calibrated
probabilities" — the System One contract that TypeSafe's Jev popularized.

ecoreasoner keeps the corpus, the L0–L3 pair batteries, and the research log.
reverse-jev keeps the decision readout, the RLCD-lite trainer, the
SDK-compatible server, and the calibration harness that compares our model
against the real Jev API on identical questions.

Reference implementations studied:
- Laya (convaiinnovations/laya): ModernBERT-large + option-marker scorer head,
  REINFORCE over Gaussian-perturbed distributions, reward = log + 0.75·spherical.
- Kev (jaredpalmer/kev): LoRA + pointer head on Qwen3.5, `/v1/systemone`
  compatible, temperature ~2.1–2.4 fitted on dev.
- OpenJev (razorback16/openjev): DiffusionGemma denoising of answer slots.
