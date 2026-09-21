"""reverse_jev.serve — /v1/systemone server compatible with the TypeSafe SDK.

  REVJEV_CKPT=runs/ckpt/model.pt REVJEV_TOKENIZER=GSAI-ML/LLaDA-8B-Instruct \
    uvicorn reverse_jev.serve:app --port 8009

Then point the official SDK at it:

  TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009")

Optional REVJEV_TEMPERATURE applies a fitted temperature to every answer.
The server binds locally; it has no auth — keep it behind localhost.
"""
import os
import time

import torch

from .model import load_backbone
from .readout import answer_questions

app = None  # set below


def _build():
    from fastapi import FastAPI
    from pydantic import BaseModel

    ckpt = os.environ["REVJEV_CKPT"]
    tok_name = os.environ.get("REVJEV_TOKENIZER", "GSAI-ML/LLaDA-8B-Instruct")
    cfg_path = os.environ.get("REVJEV_CONFIG")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = None
    if cfg_path:
        import yaml
        cfg = yaml.safe_load(open(cfg_path)).get("model", {})
    model = load_backbone(ckpt, cfg, device=device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)
    temperature = float(os.environ.get("REVJEV_TEMPERATURE", "1.0"))
    max_len = int(os.environ.get("REVJEV_MAX_LEN", "768"))

    api = FastAPI(title="reverse-jev", version="0.1.0")

    class Req(BaseModel):
        state: object
        model: str = "revjev-latest"
        questions: dict

    @api.get("/v1/models")
    def models():
        return {"model": "revjev-latest", "checkpoint": ckpt,
                "temperature": temperature, "device": device}

    @api.post("/v1/systemone")
    def systemone(req: Req):
        state = req.state if isinstance(req.state, str) else str(req.state)
        t0 = time.time()
        answers = answer_questions(model, tok, state, req.questions,
                                   temperature=temperature, max_len=max_len,
                                   device=device)
        n_in = len(tok.encode(state))
        return {"model": "revjev-latest", "answers": answers,
                "usage": {"input_tokens": n_in,
                          "output_tokens": len(answers) * 8},
                "latency_ms": round((time.time() - t0) * 1000, 1)}

    return api


app = _build()
