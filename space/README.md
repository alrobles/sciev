---
title: Sciev — Typed Decisions in One Forward Pass
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Typed decision heads (choice/noul/score) on a frozen LLaDA-8B backbone — one forward pass per question, no generation
---

# Sciev — Typed Decisions in One Forward Pass

Unstructured `state` + typed questions (`choice` / `noul` / `score`) →
structured decisions with probabilities. No text generation.

v0.3.0 ships anti-leakage `noul`/`score` heads (`alp25`): under corrupted or
missing evidence they collapse to the rejection class instead of exploiting
option-side cues. The **ablate evidence** toggle on the Verify tab shows this
live. `choice` remains the v0.2 head with documented leakage.

- Model + heads: [alrobles/sciev](https://huggingface.co/alrobles/sciev)
- Code: [github.com/alrobles/sciev](https://github.com/alrobles/sciev) (`pip install sciev`)
- Manuscript: [sciev.org/static/sciev-paper.pdf](https://sciev.org/static/sciev-paper.pdf)
- Live API (gated): [sciev.ecoseek.org](https://sciev.ecoseek.org)
