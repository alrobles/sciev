"""Sciev demo Space — typed decisions in one forward pass.

Loads the released heads from the model repo (alrobles/sciev, release/)
with the loadable manifest, on a frozen LLaDA-8B-Instruct backbone.
Designed for ZeroGPU; runs on CPU locally (slow) for smoke tests.
"""

import json
import os

import gradio as gr
import torch
from huggingface_hub import snapshot_download

try:
    import spaces
except ImportError:  # local dev without the spaces runtime
    class spaces:
        @staticmethod
        def GPU(duration=60):
            return lambda fn: fn


MODEL_REPO = os.environ.get("SCIEV_MODEL_REPO", "alrobles/sciev")

_engine = None


@spaces.GPU(duration=120)
def _engine():
    global _engine
    if _engine is None:
        from sciev.inference import DecisionEngine

        root = snapshot_download(MODEL_REPO, allow_patterns="release/*")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _engine = DecisionEngine.from_manifest(
            os.path.join(root, "release", "manifest.json"), device=device)
    return _engine


@spaces.GPU(duration=60)
def _answer(state, questions):
    return _engine().answer(state, questions)


def _noul_state(passage, claim):
    parts = [p for p in (passage.strip(), f"Claim: {claim.strip()}") if p]
    return "\n".join(parts)


def verify(passage, claim, ablate):
    if not claim.strip():
        raise gr.Error("A claim is required.")
    question = {"q1": {"type": "noul",
                       "instructions": "Is the claim supported by the passage?"}}
    full = _answer(_noul_state(passage, claim), question)["answers"]["q1"]
    out = {"with evidence": full["probabilities"]}
    raw = {"with evidence": full}
    if ablate:
        empty = _answer(f"Claim: {claim.strip()}", question)["answers"]["q1"]
        out["empty evidence"] = empty["probabilities"]
        raw["empty evidence"] = empty
    return out, raw


def choose(state, instructions, options_text):
    options = [line.strip() for line in options_text.splitlines() if line.strip()]
    if not 2 <= len(options) <= 8:
        raise gr.Error("Provide 2-8 options, one per line.")
    question = {"q1": {"type": "choice",
                       "instructions": instructions.strip() or "Which answer is correct?",
                       "criteria": {name: None for name in options}}}
    ans = _answer(state, question)["answers"]["q1"]
    return ans["probabilities"], ans


def grade(passage, claim):
    if not claim.strip():
        raise gr.Error("A claim is required.")
    question = {"q1": {"type": "score", "instructions": "Rate the claim.",
                       "criteria": ["unrelated answer", "related but wrong answer",
                                    "correct answer"]}}
    ans = _answer(_noul_state(passage, claim), question)["answers"]["q1"]
    labels = {ans["legend"][k]: p for k, p in ans["probabilities"].items()}
    return labels, ans


SCIFACT_PASSAGE = (
    "Title: Increased Microerythrocyte Count in Homozygous α+-Thalassaemia Contributes to "
    "Protection against Severe Malarial Anaemia\n"
    "Abstract: BACKGROUND The heritable haemoglobinopathy alpha(+)-thalassaemia is caused by "
    "the reduced synthesis of alpha-globin chains... Individuals homozygous for "
    "alpha(+)-thalassaemia have microcytosis and an increased erythrocyte count..."
)
SCIFACT_CLAIM = ("A high microerythrocyte count raises vulnerability to severe anemia "
                 "in homozygous alpha(+)-thalassemia trait subjects.")

GPQA_STATE = (
    "Question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and "
    "10^-8 sec, respectively. We want to clearly distinguish these two energy levels. "
    "Which one of the following options could be their energy difference so that they can "
    "be clearly resolved?")

DESCRIPTION = """\
**Sciev** answers typed questions about unstructured state in **one forward pass per
question** — no text generation, no output parsing. Heads on a frozen
[LLaDA-8B](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) masked-diffusion backbone.

v0.3.0 ships anti-leakage `noul`/`score` heads: corrupted or missing evidence collapses
to the rejection class instead of pattern-matching option-side cues. Toggle *ablate
evidence* on the Verify tab to see it live.

`choice` remains the v0.2 head with documented option-side leakage — use it as a baseline,
not a judge. `confidence` is top-probability concentration, not P(correct).

[model card](https://huggingface.co/alrobles/sciev) ·
[code](https://github.com/alrobles/sciev) ·
[paper](https://sciev.org/static/sciev-paper.pdf?v=030) · `pip install sciev`
"""

with gr.Blocks(title="Sciev — typed decisions, one forward pass") as demo:
    gr.Markdown(f"# Sciev\n\n{DESCRIPTION}")

    with gr.Tab("Verify (noul)"):
        passage = gr.Textbox(label="Passage / evidence", lines=8,
                             placeholder="Title: ...\nAbstract: ...", value=SCIFACT_PASSAGE)
        claim = gr.Textbox(label="Claim", value=SCIFACT_CLAIM)
        ablate = gr.Checkbox(label="also evaluate with empty evidence "
                                   "(anti-leakage control)")
        btn = gr.Button("Verify", variant="primary")
        probs = gr.Label(label="P(supported)")
        raw = gr.JSON(label="raw answer")
        btn.click(verify, [passage, claim, ablate], [probs, raw])

    with gr.Tab("Choose (choice)"):
        cstate = gr.Textbox(label="State / question", lines=6, value=GPQA_STATE)
        cinstr = gr.Textbox(label="Instructions", value="Which answer is correct?")
        copts = gr.Textbox(label="Options (one per line)", lines=5,
                           value="10^-4 eV\n10^-9 eV\n10^-11 eV\n10^-8 eV")
        cbtn = gr.Button("Choose", variant="primary")
        cprobs = gr.Label(label="P(option)")
        craw = gr.JSON(label="raw answer")
        cbtn.click(choose, [cstate, cinstr, copts], [cprobs, craw])

    with gr.Tab("Grade (score)"):
        spassage = gr.Textbox(label="Passage / evidence", lines=8, value=SCIFACT_PASSAGE)
        sclaim = gr.Textbox(label="Claim / answer to grade", value=SCIFACT_CLAIM)
        sbtn = gr.Button("Grade", variant="primary")
        sprobs = gr.Label(label="P(level)")
        sraw = gr.JSON(label="raw answer")
        sbtn.click(grade, [spassage, sclaim], [sprobs, sraw])

    gr.Markdown(
        "First call downloads ~16 GB of weights and loads the backbone — expect up to a "
        "minute. One forward pass per question after that.")


if __name__ == "__main__":
    demo.launch()
