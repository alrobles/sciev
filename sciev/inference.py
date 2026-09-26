import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .model import AttnPoolHead, DecisionHead, HFBackbone


QUESTION_TYPES = ("choice", "noul", "score")
CONFIDENCE_SEMANTICS = "normalized_top_concentration; not P(correct); acceptance uses max_probability"
_CURRENT_ENCODING = object()


class InputValidationError(ValueError):
    pass


class UnknownModelError(InputValidationError):
    pass


class MissingHeadError(InputValidationError):
    pass


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value):
        raise ValueError("model identifier must be a nonempty name, not a filesystem path")
    return value


@dataclass(frozen=True)
class HeadSettings:
    head_kind: str = "mlp"
    mode: str = "spanpool"
    layers: tuple[int, ...] = (-1,)
    temperature: float = 1.0
    canonical_order: bool = True

    def __post_init__(self):
        if self.head_kind not in ("mlp", "attnpool"):
            raise ValueError("head_kind must be mlp or attnpool")
        if self.mode not in ("spanpool", "marker"):
            raise ValueError("mode must be spanpool or marker")
        if not isinstance(self.layers, (tuple, list)) or not self.layers:
            raise ValueError("layers must contain at least one integer index")
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in self.layers):
            raise ValueError("layers must contain integer indices")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must not contain duplicate indices")
        object.__setattr__(self, "layers", tuple(self.layers))
        if self.head_kind == "attnpool" and self.mode != "spanpool":
            raise ValueError("attnpool heads require spanpool mode")
        if self.head_kind == "mlp" and self.layers != (-1,):
            raise ValueError("mlp heads read only the final layer; layers must be [-1]")
        if (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError("temperature must be positive and finite")
        if not isinstance(self.canonical_order, bool):
            raise ValueError("canonical_order must be boolean")


def _limits(max_ctx, max_opt, max_questions, seq_len):
    return {
        "choice": {"min_options": 2, "max_options": 255},
        "score": {"min_levels": 2, "max_levels": 10},
        "noul": {"options": 2, "criteria_keys": ["true", "false"]},
        "max_context_tokens": max_ctx,
        "max_option_tokens": max_opt,
        "max_sequence_tokens": seq_len,
        "max_questions": max_questions,
        "overflow": "error",
    }


def _validate_request(state, questions, model, model_id, max_questions):
    if model is not None and model != model_id:
        raise UnknownModelError(f"unknown model; available model is {model_id}")
    if not isinstance(state, (str, dict, list)):
        raise InputValidationError("state must be a string, object, or array")
    if not isinstance(questions, Mapping) or not 1 <= len(questions) <= max_questions:
        raise InputValidationError(f"questions must contain 1..{max_questions} entries")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid.strip():
            raise InputValidationError("question ids must be nonempty strings")
        if not isinstance(question, Mapping) or question.get("type") not in QUESTION_TYPES:
            raise InputValidationError("question type must be choice, noul, or score")
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise InputValidationError("choice requires 2..255 named options")
            if any(not isinstance(key, str) or not key.strip() for key in criteria):
                raise InputValidationError("choice option names must be nonempty strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise InputValidationError("score requires 2..10 ordered levels")
        elif criteria is not None and (not isinstance(criteria, dict)
                                       or set(criteria) != {"true", "false"}):
            raise InputValidationError("noul criteria must contain exactly true and false")


def _probabilities(logits, count, temperature):
    if not isinstance(logits, torch.Tensor) or logits.shape != (count,):
        raise RuntimeError("head returned an invalid logit shape")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("head returned nonfinite logits")
    values = logits.to(dtype=torch.float64).cpu().tolist()
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _format_answer(row, question, probabilities, *, selected_index=None):
    kind = row["kind"]
    maximum = max(probabilities)
    count = len(probabilities)
    concentration = max(0.0, min(1.0, (maximum - 1.0 / count) / (1.0 - 1.0 / count)))
    answer = {"type": kind, "confidence": concentration,
              "concentration": concentration, "max_probability": maximum}
    if kind == "choice":
        keys = row["option_keys"]
        if selected_index is None:
            selected_index = min(range(count), key=lambda index: (-probabilities[index], keys[index]))
        answer.update(choice=keys[selected_index], probabilities=dict(zip(keys, probabilities)))
    elif kind == "noul":
        keys = [str(key).lower() for key in row["option_keys"]]
        positive = [i for i, key in enumerate(keys) if key in ("true", "yes")]
        negative = [i for i, key in enumerate(keys) if key in ("false", "no")]
        if len(positive) != 1 or len(negative) != 1:
            raise RuntimeError("noul encoding must identify yes and no options")
        yes, no = probabilities[positive[0]], probabilities[negative[0]]
        answer.update(noul=yes, probabilities={"true": yes, "false": no})
    else:
        answer.update(score=math.fsum(i * p for i, p in enumerate(probabilities)),
                      legend={str(i): level for i, level in enumerate(question["criteria"])},
                      probabilities={str(i): p for i, p in enumerate(probabilities)})
    return answer


def _layer_count(backbone):
    if hasattr(backbone, "blocks"):
        return len(backbone.blocks)
    config = getattr(getattr(backbone, "hf", None), "config", None)
    count = getattr(config, "num_hidden_layers", None)
    return count + 1 if isinstance(count, int) and count > 0 else None


def _validate_head(backbone, head, settings):
    if not isinstance(head, torch.nn.Module):
        raise ValueError("heads must be torch modules")
    actual_kind = "attnpool" if isinstance(head, AttnPoolHead) else "mlp"
    if actual_kind != settings.head_kind:
        raise ValueError("head class does not match head_kind")
    if actual_kind == "attnpool" and head.n_layers != len(settings.layers):
        raise ValueError("head layer count does not match layers")
    layer_count = _layer_count(backbone)
    if layer_count is not None and any(not -layer_count <= layer < layer_count for layer in settings.layers):
        raise ValueError("head layer index is outside the shared backbone")
    dim = getattr(getattr(backbone, "tok_emb", None), "embedding_dim", None)
    if isinstance(head, (AttnPoolHead, DecisionHead)) and dim is not None:
        if head.net[-1].in_features != dim:
            raise ValueError("head hidden dimension does not match the shared backbone")


class DecisionEngine:
    def __init__(self, backbone, tokenizer, heads, settings=None, *, model_id="sciev-local",
                 device="cpu", max_ctx=640, max_opt=120, max_questions=128,
                 bundle_encoding=_CURRENT_ENCODING, allow_legacy_encoding=False):
        from .decisions import ENCODING_VERSION

        self.model_id = _identifier(model_id)
        self.max_ctx = _positive_int(max_ctx, "max_ctx")
        self.max_opt = _positive_int(max_opt, "max_opt")
        self.max_questions = _positive_int(max_questions, "max_questions")
        self.encoding = ENCODING_VERSION
        self.bundle_encoding = ENCODING_VERSION if bundle_encoding is _CURRENT_ENCODING else bundle_encoding
        _validate_encoding(self.bundle_encoding, allow_legacy_encoding)
        if not isinstance(backbone, torch.nn.Module):
            raise ValueError("a shared torch backbone is required")
        if not callable(getattr(tokenizer, "encode", None)):
            raise ValueError("a tokenizer with encode() is required")
        if not isinstance(heads, Mapping) or not heads or set(heads) - set(QUESTION_TYPES):
            raise ValueError("heads must be a nonempty mapping of choice, noul, or score heads")
        settings = {} if settings is None else settings
        if not isinstance(settings, Mapping) or set(settings) - set(heads):
            raise ValueError("settings must describe only available heads")
        self.settings = {}
        for kind, head in heads.items():
            configured = settings.get(kind)
            if configured is None:
                configured = HeadSettings(head_kind="attnpool" if isinstance(head, AttnPoolHead) else "mlp")
            elif isinstance(configured, Mapping):
                configured = HeadSettings(**configured)
            if not isinstance(configured, HeadSettings):
                raise ValueError("settings must be HeadSettings or settings mappings")
            _validate_head(backbone, head, configured)
            self.settings[kind] = configured
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device).eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.heads = {kind: head.to(self.device).eval().requires_grad_(False) for kind, head in heads.items()}
        self._lock = threading.RLock()

    def _calibration(self):
        legacy = self.bundle_encoding != self.encoding
        return {
            "status": "legacy_unverified" if legacy else "encoding_compatible_unverified",
            "verified": False,
            "temperatures": {kind: self.settings[kind].temperature for kind in QUESTION_TYPES if kind in self.heads},
            "note": ("Legacy temperatures are unverified for systemone-v2 text inputs."
                     if legacy else "Matching encoding alone does not validate calibration or P(correct)."),
        }

    def metadata(self):
        return {"readout": "r2", "encoding": self.encoding, "bundle_encoding": self.bundle_encoding,
                "calibration": self._calibration(), "confidence_semantics": CONFIDENCE_SEMANTICS}

    def model_info(self):
        return {"model": self.model_id, **self.metadata(),
                "available_types": [kind for kind in QUESTION_TYPES if kind in self.heads],
                "heads": {kind: asdict(self.settings[kind]) for kind in QUESTION_TYPES if kind in self.heads},
                "limits": _limits(self.max_ctx, self.max_opt, self.max_questions, self.backbone.seq_len)}

    def answer(self, state, questions, *, model=None):
        from .decisions import (
            decision_logits, decision_prediction, decision_probabilities,
            encode_question, prepare_decision,
        )

        _validate_request(state, questions, model, self.model_id, self.max_questions)
        with self._lock, torch.inference_mode():
            encoded = []
            for qid, question in questions.items():
                kind = question["type"]
                if kind not in self.heads:
                    raise MissingHeadError(f"no {kind} head is available for model {self.model_id}")
                settings = self.settings[kind]
                try:
                    row = encode_question(self.tokenizer, state, question, max_ctx=self.max_ctx,
                                          max_opt=self.max_opt, overflow="error")
                    if not row["ctx"] or not row["opts"] or any(not option for option in row["opts"]):
                        raise ValueError("context and options must contain tokens")
                    prepare_decision(self.backbone, row["ctx"], row["opts"], mode=settings.mode,
                                     canonical=settings.canonical_order, strict=True)
                except (ValueError, TypeError) as exc:
                    raise InputValidationError(f"question {qid}: {exc}") from exc
                encoded.append((qid, question, row, settings))
            answers, input_tokens = {}, 0
            for qid, question, row, settings in encoded:
                try:
                    logits, prepared = decision_logits(
                        self.backbone, self.heads[row["kind"]], row["ctx"], row["opts"], self.device,
                        mode=settings.mode, layers=settings.layers,
                        canonical=settings.canonical_order, strict=True)
                    probabilities = decision_probabilities(logits, settings.temperature)
                    selected_index = decision_prediction(logits, prepared)
                except (ValueError, FloatingPointError) as exc:
                    raise RuntimeError("R2 inference did not produce finite, correctly shaped logits") from exc
                answers[qid] = _format_answer(row, question, probabilities, selected_index=selected_index)
                input_tokens += len(prepared.ids)
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0, "forward_passes": len(encoded)},
                "metadata": self.metadata()}

    @classmethod
    def from_manifest(cls, manifest_path, *, device=None, allow_legacy_encoding=False,
                      tokenizer=None, backbone_factory=None, tokenizer_factory=None):
        path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        backbone_name = manifest.get("backbone")
        if not isinstance(backbone_name, str) or not backbone_name.strip():
            raise ValueError("manifest requires a backbone identifier")
        release = manifest.get("release")
        if not isinstance(release, str) or not release.strip():
            raise ValueError("manifest requires a release identifier")
        model_id = _identifier(manifest.get("model_id", manifest.get("model", f"sciev-{release}")))
        encoding = manifest.get("encoding")
        _validate_encoding(encoding, allow_legacy_encoding)
        seq_len = _positive_int(manifest.get("seq_len", 2048), "seq_len")
        max_ctx = _positive_int(manifest.get("max_ctx", 640), "max_ctx")
        max_opt = _positive_int(manifest.get("max_opt", 120), "max_opt")
        max_questions = _positive_int(manifest.get("max_questions", 128), "max_questions")
        dtype = manifest.get("dtype", "bfloat16")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        adapter = _adapter_path(manifest.get("lora_adapter"), path.parent)
        specs = manifest.get("heads")
        if not isinstance(specs, dict) or not specs or set(specs) - set(QUESTION_TYPES):
            raise ValueError("manifest heads must declare available choice, noul, or score heads")
        if tokenizer is not None and tokenizer_factory is not None:
            raise ValueError("supply tokenizer or tokenizer_factory, not both")
        settings, files = {}, {}
        for kind, spec in specs.items():
            required = {"head_kind", "mode", "layers", "temperature", "canonical_order"}
            if not isinstance(spec, dict) or not required.issubset(spec):
                raise ValueError(f"{kind} head requires explicit inference settings")
            settings[kind] = HeadSettings(**{key: spec[key] for key in required})
            filename = spec.get("file")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{kind} head requires a checkpoint file")
            files[kind] = (path.parent / Path(filename).expanduser()).resolve()
            expected = spec.get("sha256")
            if "sha256" in spec:
                if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                    raise ValueError(f"{kind} head has an invalid SHA256 checksum")
                digest = hashlib.sha256()
                with files[kind].open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected.lower():
                    raise ValueError(f"{kind} head SHA256 checksum mismatch")
            elif not files[kind].is_file():
                raise ValueError(f"{kind} head checkpoint file does not exist")
        heads, dimensions = {}, set()
        config = {"hf_backbone": backbone_name, "encoding": encoding, "seq_len": seq_len,
                  "max_ctx": max_ctx, "max_opt": max_opt, "dtype": dtype}
        for kind, file in files.items():
            try:
                checkpoint = torch.load(file, weights_only=True, map_location="cpu")
            except Exception as exc:
                raise ValueError(f"{kind} checkpoint failed weights-only deserialization") from exc
            _validate_checkpoint(checkpoint, specs[kind], settings[kind], config, adapter, path.parent,
                                 allow_legacy_encoding=allow_legacy_encoding)
            head = _load_head(checkpoint["head"], settings[kind])
            heads[kind] = head
            dimensions.add(head.net[-1].in_features)
        if len(dimensions) != 1:
            raise ValueError("heads require different backbone hidden dimensions")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        factory = backbone_factory if backbone_factory is not None else HFBackbone
        backbone = factory(backbone_name, device=device, dtype=dtype, seq_len=seq_len, lora_adapter=adapter)
        if getattr(getattr(backbone, "tok_emb", None), "embedding_dim", None) not in dimensions:
            raise ValueError("checkpoint head dimensions do not match the shared backbone")
        for kind, head in heads.items():
            _validate_head(backbone, head, settings[kind])
        if tokenizer is None:
            if tokenizer_factory is None:
                from transformers import AutoTokenizer
                tokenizer_factory = AutoTokenizer.from_pretrained
            tokenizer = tokenizer_factory(backbone_name)
        return cls(backbone, tokenizer, heads, settings, model_id=model_id, device=device,
                   max_ctx=max_ctx, max_opt=max_opt, max_questions=max_questions,
                   bundle_encoding=encoding, allow_legacy_encoding=allow_legacy_encoding)


def _validate_encoding(encoding, allow_legacy):
    from .decisions import ENCODING_VERSION

    if encoding is not None and (not isinstance(encoding, str) or not encoding.strip()):
        raise ValueError("bundle encoding must be a nonempty string when declared")
    if not isinstance(allow_legacy, bool):
        raise ValueError("allow_legacy_encoding must be boolean")
    if encoding != ENCODING_VERSION and not allow_legacy:
        raise ValueError("absent or legacy encoding requires explicit allow_legacy_encoding=True; calibration is unverified")


def _adapter_path(value, parent):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("LoRA adapter must be an explicit nonempty path or null")
    return str((parent / Path(value).expanduser()).resolve())


def _checkpoint_sources(checkpoint, spec):
    sources = [checkpoint]
    for key in ("meta", "inference", "inference_settings", "settings"):
        value = checkpoint.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError("checkpoint metadata must be a mapping")
            sources.append(value)
            for nested_key in ("inference", "inference_settings", "settings"):
                nested = value.get(nested_key)
                if nested is not None:
                    if not isinstance(nested, Mapping):
                        raise ValueError("checkpoint inference metadata must be a mapping")
                    sources.append(nested)
    return sources + [spec]


def _validate_checkpoint(checkpoint, spec, settings, config, adapter, parent, *, allow_legacy_encoding=False):
    from .decisions import ENCODING_VERSION

    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("head"), Mapping):
        raise ValueError("checkpoint must contain a head state dictionary")
    if checkpoint.get("hf_backbone") != config["hf_backbone"] or checkpoint.get("model") is not None:
        raise ValueError("checkpoint hf_backbone does not match the shared HF backbone")
    sources = _checkpoint_sources(checkpoint, spec)
    checkpoint_sources = sources[:-1]
    for required in ("head_kind", "n_layers"):
        if not any(required in source for source in checkpoint_sources):
            raise ValueError(f"checkpoint is missing {required}")
    if not any("mode" in source or "r2_mode" in source for source in checkpoint_sources):
        raise ValueError("checkpoint is missing its decision mode")
    declared_adapters = [source["lora_adapter"] for source in checkpoint_sources if "lora_adapter" in source]
    if not declared_adapters:
        declared_adapters = [None]
    if "lora_adapter" in spec:
        declared_adapters.append(spec["lora_adapter"])
    if any(_adapter_path(value, parent) != adapter for value in declared_adapters):
        raise ValueError("checkpoint LoRA adapter does not match bundle configuration")
    legacy_mlp_count = (allow_legacy_encoding and config["encoding"] != ENCODING_VERSION
                        and settings.head_kind == "mlp")
    for source_index, source in enumerate(sources):
        for name, expected in config.items():
            if name in source and source[name] != expected:
                raise ValueError(f"checkpoint {name} does not match bundle configuration")
        if "head_kind" in source and source["head_kind"] != settings.head_kind:
            raise ValueError("checkpoint head_kind does not match manifest")
        if "n_layers" in source:
            count = _positive_int(source["n_layers"], "n_layers")
            if count != len(settings.layers) and not (legacy_mlp_count and source_index == 0):
                raise ValueError("checkpoint layer count does not match manifest")
        for key in ("mode", "r2_mode"):
            if key in source:
                mode = source[key]
                if not isinstance(mode, str) or mode.removeprefix("r2_") != settings.mode:
                    raise ValueError("checkpoint mode does not match manifest")
        for key in ("layers", "layers_list", "r2_layers"):
            if key in source:
                layers = source[key]
                if isinstance(layers, str):
                    try:
                        layers = [int(value) for value in layers.split(",")]
                    except ValueError as exc:
                        raise ValueError("checkpoint layers must contain integer indices") from exc
                if (not isinstance(layers, (list, tuple))
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in layers)
                        or tuple(layers) != settings.layers):
                    raise ValueError("checkpoint layers do not match manifest")
        for key in ("canonical", "canonical_order"):
            if key in source and (not isinstance(source[key], bool) or source[key] != settings.canonical_order):
                raise ValueError("checkpoint canonical_order does not match manifest")
        if "temperature" in source:
            value = source["temperature"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value != settings.temperature):
                raise ValueError("checkpoint temperature does not match manifest")


def _load_head(state, settings):
    if not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                        for key, value in state.items()):
        raise ValueError("head state must contain named tensors")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("head state contains nonfinite tensors")
    weight = state.get("net.3.weight")
    if weight is None or weight.ndim != 2 or weight.shape[0] != 1 or weight.shape[1] < 1:
        raise ValueError("head state has an invalid scorer dimension")
    head = (AttnPoolHead(weight.shape[1], n_layers=len(settings.layers))
            if settings.head_kind == "attnpool" else DecisionHead(weight.shape[1]))
    try:
        head.load_state_dict(state, strict=True)
    except (ValueError, RuntimeError) as exc:
        raise ValueError("head state does not strictly match declared settings") from exc
    return head.eval().requires_grad_(False)
