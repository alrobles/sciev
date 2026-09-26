"""sciev.serve — /v1/systemone server compatible with the TypeSafe SDK.

  REVJEV_CKPT=runs/ckpt/model.pt REVJEV_TOKENIZER=GSAI-ML/LLaDA-8B-Instruct \
    uvicorn sciev.serve:app --port 8009

Then point the official SDK at it:

  TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009")

Optional REVJEV_TEMPERATURE applies a fitted temperature to every answer.
The server binds locally; it has no auth — keep it behind localhost.
"""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from .inference import (
    CONFIDENCE_SEMANTICS, QUESTION_TYPES, DecisionEngine, HeadSettings,
    InputValidationError, UnknownModelError, _format_answer, _limits,
    _positive_int, _probabilities, _validate_request,
)

app = None  # set below
logger = logging.getLogger(__name__)
StructuredText = str | dict[str, JsonValue] | list[JsonValue]
NonemptyString = Annotated[str, Field(min_length=1)]


class _Question(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)
    instructions: StructuredText


class ChoiceQuestion(_Question):
    type: Literal["choice"]
    criteria: dict[NonemptyString, StructuredText | None] = Field(min_length=2, max_length=255)


class NoulQuestion(_Question):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], StructuredText] | None = None

    @field_validator("criteria")
    @classmethod
    def both_criteria(cls, value):
        if value is not None and set(value) != {"true", "false"}:
            raise ValueError("noul criteria must contain exactly true and false")
        return value


class ScoreQuestion(_Question):
    type: Literal["score"]
    criteria: list[StructuredText] = Field(min_length=2, max_length=10)


Question = Annotated[ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    state: StructuredText
    model: NonemptyString | None = None
    questions: dict[NonemptyString, Question] = Field(min_length=1)


class _LegacyR1Engine:
    def __init__(self, backbone, tokenizer, temperature, max_len, device):
        self.model_id = "revjev-r1-legacy"
        self.backbone = backbone.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.temperature = HeadSettings(temperature=temperature).temperature
        self.max_len = min(_positive_int(max_len, "max_len"), backbone.seq_len)
        if self.max_len < 2:
            raise ValueError("legacy max_len must leave room for input and a mask token")
        self.max_questions = 128
        self.device = device
        self._lock = threading.RLock()

    @classmethod
    def from_environment(cls):
        from .model import load_backbone
        from transformers import AutoTokenizer

        logger.warning("REVJEV_CKPT explicitly selects the R1 legacy readout, not R2 specialist heads.")
        config = None
        config_path = os.environ.get("REVJEV_CONFIG")
        if config_path:
            import yaml
            config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")).get("model", {})
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone = load_backbone(os.environ["REVJEV_CKPT"], config, device=device)
        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("REVJEV_TOKENIZER", "GSAI-ML/LLaDA-8B-Instruct"))
        return cls(backbone, tokenizer, float(os.environ.get("REVJEV_TEMPERATURE", "1.0")),
                   int(os.environ.get("REVJEV_MAX_LEN", "768")), device)

    def metadata(self):
        return {
            "readout": "r1", "legacy": True, "encoding": "r1-token-slot-legacy",
            "calibration": {"status": "legacy_unverified", "verified": False,
                            "temperatures": {kind: self.temperature for kind in QUESTION_TYPES}},
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "limitations": ["R1 does not score choice descriptions or score rubric descriptions.",
                            "R1 does not use the released R2 specialist heads."],
        }

    def model_info(self):
        return {"model": self.model_id, **self.metadata(), "available_types": list(QUESTION_TYPES),
                "limits": _limits(self.max_len - 1, 120, self.max_questions, self.max_len)}

    def answer(self, state, questions, *, model=None):
        from .decisions import encode_question, render_value
        from .readout import build_sequence, noul_id_sets, option_id_sets, r1_logits

        _validate_request(state, questions, model, self.model_id, self.max_questions)
        with self._lock, torch.inference_mode():
            encoded = []
            for qid, question in questions.items():
                try:
                    row = encode_question(self.tokenizer, state, question, max_ctx=self.max_len - 1,
                                          max_opt=120, overflow="error")
                    ids, mask_position = build_sequence(
                        self.tokenizer, render_value(state, "state", allow_empty=True),
                        render_value(question["instructions"], "instructions"),
                        self.backbone.mask_id, max_len=2 ** 63 - 1)
                    if not ids[:-1] or len(ids) > self.max_len:
                        raise ValueError("legacy input exceeds its token budget or contains no context")
                    if any(token < 0 or token >= self.backbone.tok_emb.num_embeddings for token in ids):
                        raise ValueError("legacy input token is outside the model vocabulary")
                    if row["kind"] == "noul":
                        option_sets = noul_id_sets(self.tokenizer)
                    else:
                        option_sets = [option_id_sets(self.tokenizer, key) for key in row["option_keys"]]
                    if any(not option or any(token < 0 or token >= self.backbone.vocab for token in option)
                           for option in option_sets):
                        raise ValueError("legacy option token is outside the model vocabulary")
                    if len({tuple(option) for option in option_sets}) != len(option_sets):
                        raise ValueError("legacy first-token option sets are indistinguishable")
                except (ValueError, TypeError) as exc:
                    raise InputValidationError(f"question {qid}: {exc}") from exc
                encoded.append((qid, question, row, ids, mask_position, option_sets))
            answers, input_tokens = {}, 0
            for qid, question, row, ids, mask_position, option_sets in encoded:
                logits = r1_logits(self.backbone, torch.tensor(ids, dtype=torch.long, device=self.device),
                                   mask_position, option_sets, temperature=1.0)
                probabilities = _probabilities(logits, len(option_sets), self.temperature)
                answers[qid] = _format_answer(row, question, probabilities)
                input_tokens += len(ids)
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0, "forward_passes": len(encoded)},
                "metadata": self.metadata()}


def _engine_from_environment():
    if "SCIEV_MANIFEST" in os.environ:
        path = os.environ["SCIEV_MANIFEST"]
        if not path.strip():
            raise ValueError("SCIEV_MANIFEST must name a manifest")
        legacy_encoding = os.environ.get("SCIEV_ALLOW_LEGACY_ENCODING", "0")
        if legacy_encoding not in ("0", "1"):
            raise ValueError("SCIEV_ALLOW_LEGACY_ENCODING must be 0 or 1")
        return DecisionEngine.from_manifest(path, device=os.environ.get("SCIEV_DEVICE"),
                                            allow_legacy_encoding=legacy_encoding == "1")
    if os.environ.get("REVJEV_CKPT"):
        return _LegacyR1Engine.from_environment()
    raise ValueError("configure SCIEV_MANIFEST for R2 or REVJEV_CKPT for explicit R1 legacy serving")


def create_app(engine=None):
    @asynccontextmanager
    async def lifespan(api):
        logger.warning("Local-only server has no authentication; keep it behind localhost or an authenticated proxy.")
        yield

    api = FastAPI(
        title="Sciev", version="0.1.1", lifespan=lifespan,
        description=("Typed non-generative decisions. confidence is normalized-top concentration, not P(correct). "
                     "max_probability is separate. Calibration status and encoding are reported explicitly. "
                     "Local-only: this server has no authentication."))
    api.state.engine = engine
    api.state.engine_lock = threading.Lock()
    api.state.initialization_failed = False

    def get_engine():
        if api.state.engine is None:
            with api.state.engine_lock:
                if api.state.engine is None and not api.state.initialization_failed:
                    try:
                        api.state.engine = _engine_from_environment()
                    except Exception:
                        api.state.initialization_failed = True
                        logger.exception("Configured decision model could not be initialized")
                if api.state.engine is None:
                    raise HTTPException(status_code=503, detail="Decision model is unavailable; check server configuration and logs.")
        return api.state.engine

    @api.get("/v1/models")
    def models():
        return get_engine().model_info()

    @api.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        current = get_engine()
        started = time.perf_counter()
        questions = {qid: question.model_dump() for qid, question in req.questions.items()}
        try:
            result = current.answer(req.state, questions, model=req.model)
        except UnknownModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {**result, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}

    return api


def _build():
    return create_app()


app = create_app()
