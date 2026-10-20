# Try Sciev

Sciev answers **typed questions** about an unstructured `state` — no
generation, one backbone forward pass per question, calibrated
probabilities. Three decision types:

| type | what it does | options |
|---|---|---|
| `choice` | pick the best option | 2–4 options you provide |
| `noul` | verify a claim against evidence | `true` / `false` |
| `score` | ordinal grading | levels you provide |

## 1. Live demo — HF Space (zero install)

**https://huggingface.co/spaces/alrobles/sciev**

- **Verify**: paste a passage + a claim → `true`/`false` with calibrated
  probabilities. Toggle **ablate evidence** to run the same claim with an
  empty evidence block — on the anti-leakage heads the prediction should
  collapse to the prior (that's the point: it *knows* when it has no
  evidence).
- **Choose**: state + one option per line → probability over options.
- **Grade**: passage + claim → ordinal score.

The Space runs on ZeroGPU; cold starts take ~1 min (the LLaDA-8B
backbone loads on first call). Anonymous users get a small daily quota;
logging in with a free HF account gives more.

## 2. From Python

```bash
pip install "sciev[serve]"
```

```python
from huggingface_hub import snapshot_download
from sciev.inference import DecisionEngine

root = snapshot_download("alrobles/sciev", allow_patterns="release/*")
engine = DecisionEngine.from_manifest(f"{root}/release/manifest.json")
# downloads LLaDA-8B backbone (~16 GB) on first load

result = engine.decide(
    state="Spliceosomal introns are removed via a lariat intermediate.",
    questions=[{
        "type": "noul",
        "instructions": "Is the claim supported by the evidence?",
        "criteria": {"evidence": "the passage above"},
        "label": "Introns are excised as lariats.",
    }])
print(result["max_probability"], result["choices"])
```

Head integrity is verifiable — the manifest pins SHA-256 for every head
(`fr_choice`, `alp25_noul`, `alp25_score`) and its calibration
temperature.

## 3. Serve the wire contract yourself

```bash
export SCIEV_MANIFEST=<snapshot>/release/manifest.json
uvicorn sciev.serve:app --port 8080
```

```
POST /v1/systemone   {"state": "...", "questions": [...]}
POST /v1/rank        {"state": "...", "candidates": [...]}
GET  /v1/models      served model id + head SHAs + temperatures
GET  /health
```

The app ships no auth — put it behind your own gate.

## What to expect (honest numbers)

- **SciFact (noul, external)**: 0.70 accuracy — on par with a generative
  judge ~20× its cost, on matched rows.
- **Internal battery**: 0.90+ elite accuracy; this is in-distribution —
  treat it as a calibration check, not a capability claim.
- **GPQA main (choice)**: ~0.31 parametric, ~0.72 with gold explanations.
  Graduate-level reasoning is a known limit — see the model card.
- **Anti-leakage**: trained with evidence-corruption (rho=0.25); when
  evidence is empty or shuffled, predictions collapse to the prior. The
  Space toggle lets you watch it happen.

Questions/issues: https://github.com/alrobles/sciev/issues
