import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import Mock

import pytest
import torch

from sciev.model import AttnPoolHead, DecisionHead, MdLMMoE


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [byte + 1 for byte in text.encode("utf-8")]


class TinyBackbone(MdLMMoE):
    def __init__(self, seq_len=2048, vocab=256):
        super().__init__(vocab=vocab, hidden=8, layers=2, heads=2,
                         ff_mult=1, seq_len=seq_len, n_experts=1, k=1)
        self.seen = []
        self.inference_flags = []

    def forward(self, ids, **kwargs):
        self.seen.append(ids.detach().cpu().clone())
        self.inference_flags.append(torch.is_inference_mode_enabled())
        return super().forward(ids, **kwargs)

    def hidden_layers(self, ids, layer_idx):
        self.seen.append(ids.detach().cpu().clone())
        self.inference_flags.append(torch.is_inference_mode_enabled())
        return super().hidden_layers(ids, layer_idx)


def make_engine(**kwargs):
    from sciev.inference import DecisionEngine, HeadSettings

    torch.manual_seed(41)
    backbone = kwargs.pop("backbone", TinyBackbone())
    heads = kwargs.pop("heads", {
        "choice": AttnPoolHead(8, n_layers=2),
        "noul": DecisionHead(8),
        "score": DecisionHead(8),
    })
    settings = kwargs.pop("settings", {
        "choice": HeadSettings(head_kind="attnpool", layers=(-1, -2),
                               canonical_order=True, temperature=0.8),
        "noul": HeadSettings(canonical_order=True, temperature=1.3),
        "score": HeadSettings(canonical_order=True, temperature=0.9),
    })
    return DecisionEngine(backbone, FakeTokenizer(), heads, settings,
                          model_id="synthetic-r2", device="cpu", **kwargs)


def all_questions():
    return {
        "select": {"type": "choice", "instructions": {"task": "select"},
                   "criteria": {"beta": {"evidence": [2, 1]}, "alpha": None}},
        "verify": {"type": "noul", "instructions": ["is supported?"],
                   "criteria": {"true": {"means": "supported"},
                                "false": ["contradicted", "unknown"]}},
        "rate": {"type": "score", "instructions": "rate evidence strength",
                 "criteria": ["weak", {"strength": "medium"}, ["strong"]]},
    }


def test_r2_heads_all_types_usage_and_inference_mode():
    engine = make_engine()
    calls = {kind: 0 for kind in engine.heads}
    handles = []
    for kind, head in engine.heads.items():
        def count(module, args, output, kind=kind):
            calls[kind] += 1
        handles.append(head.register_forward_hook(count))
    result = engine.answer({"evidence": ["observation"]}, all_questions())
    for handle in handles:
        handle.remove()
    assert calls == {"choice": 1, "noul": 1, "score": 1}
    assert len(engine.backbone.seen) == 3
    assert all(engine.backbone.inference_flags)
    assert result["usage"] == {
        "input_tokens": sum(ids.numel() for ids in engine.backbone.seen),
        "output_tokens": 0,
        "forward_passes": 3,
    }
    assert result["usage"]["input_tokens"] > len(FakeTokenizer().encode("observation"))
    assert not engine.backbone.training
    assert all(not p.requires_grad for p in engine.backbone.parameters())
    assert all(not head.training for head in engine.heads.values())
    assert all(not p.requires_grad for head in engine.heads.values()
               for p in head.parameters())
    answers = result["answers"]
    assert set(answers["select"]["probabilities"]) == {"beta", "alpha"}
    assert answers["select"]["choice"] in {"beta", "alpha"}
    assert answers["verify"]["noul"] == answers["verify"]["probabilities"]["true"]
    expected = sum(int(level) * prob
                   for level, prob in answers["rate"]["probabilities"].items())
    assert answers["rate"]["score"] == pytest.approx(expected)
    assert answers["rate"]["legend"] == {
        "0": "weak", "1": {"strength": "medium"}, "2": ["strong"]}
    for answer in answers.values():
        assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-12)
        assert answer["max_probability"] == max(answer["probabilities"].values())
        k = len(answer["probabilities"])
        concentration = (answer["max_probability"] - 1 / k) / (1 - 1 / k)
        assert answer["confidence"] == pytest.approx(concentration)
        assert answer["confidence"] == answer["concentration"]
    assert "concentration" in result["metadata"]["confidence_semantics"]


def test_named_probabilities_and_selection_survive_choice_permutation():
    engine = make_engine()
    question = {"type": "choice", "instructions": "pick the right option",
                "criteria": {"zebra": "first", "ant": "second", "yak": "third"}}
    first = engine.answer("facts", {"q": question})["answers"]["q"]
    question["criteria"] = dict(reversed(list(question["criteria"].items())))
    second = engine.answer("facts", {"q": question})["answers"]["q"]
    assert first["choice"] == second["choice"]
    assert first["probabilities"] == second["probabilities"]
    assert torch.equal(engine.backbone.seen[0], engine.backbone.seen[1])


def test_instruction_criteria_state_and_labels_have_correct_feature_roles():
    engine = make_engine()
    question = {"type": "score", "instructions": {"task": ["rate", "evidence"]},
                "criteria": ["weak", {"grade": "strong"}], "label": 0}
    engine.answer({"b": 2, "a": [1]}, {"first-id": question})
    question["label"] = 1
    question["qid"] = "ignored-feature"
    engine.answer({"a": [1], "b": 2}, {"different-id": question})
    assert torch.equal(engine.backbone.seen[0], engine.backbone.seen[1])
    question["instructions"] = {"task": ["rate", "quality"]}
    engine.answer({"a": [1], "b": 2}, {"q": question})
    assert not torch.equal(engine.backbone.seen[1], engine.backbone.seen[2])
    question["criteria"][1] = {"grade": "moderate"}
    engine.answer({"a": [1], "b": 2}, {"q": question})
    assert not torch.equal(engine.backbone.seen[2], engine.backbone.seen[3])


def test_choice_descriptions_and_noul_descriptions_reach_backbone():
    engine = make_engine()
    question = {"type": "choice", "instructions": "select", "criteria": {"a": None, "b": {"text": "old"}}}
    engine.answer("facts", {"q": question})
    question["criteria"]["b"] = {"text": "new"}
    engine.answer("facts", {"q": question})
    assert not torch.equal(engine.backbone.seen[0], engine.backbone.seen[1])
    question = {"type": "noul", "instructions": "is true?"}
    engine.answer("facts", {"q": question})
    question["criteria"] = {"true": ["supported"], "false": {"means": "not supported"}}
    engine.answer("facts", {"q": question})
    assert not torch.equal(engine.backbone.seen[2], engine.backbone.seen[3])


@pytest.mark.parametrize("invalid", [
    {},
    {"bad": {"type": "unknown"}},
    {"bad": {"type": "choice", "instructions": "select", "criteria": {"only": None}}},
    {"bad": {"type": "choice", "instructions": "select", "criteria": {str(i): None for i in range(256)}}},
    {"bad": {"type": "score", "criteria": ["only"]}},
    {"bad": {"type": "score", "criteria": [str(i) for i in range(11)]}},
    {"bad": {"type": "noul", "criteria": {"true": "yes"}}},
    {"bad": {"type": "noul", "criteria": {"yes": "yes", "no": "no"}}},
])
def test_invalid_questions_fail_before_any_forward(invalid):
    engine = make_engine()
    with pytest.raises(ValueError):
        engine.answer("facts", invalid)
    assert engine.backbone.seen == []


def test_later_invalid_question_does_not_partially_infer():
    engine = make_engine()
    questions = {"valid": {"type": "noul", "instructions": "is true?"},
                 "invalid": {"type": "choice", "instructions": "select", "criteria": {"only": None}}}
    with pytest.raises(ValueError):
        engine.answer("facts", questions)
    assert engine.backbone.seen == []


def test_unknown_model_and_missing_head_are_explicit():
    engine = make_engine()
    with pytest.raises(ValueError, match="model"):
        engine.answer("facts", {"q": {"type": "noul", "instructions": "verify facts"}}, model="not-installed")
    engine = make_engine(heads={"noul": DecisionHead(8)},
                         settings={"noul": {"mode": "spanpool"}})
    with pytest.raises(ValueError, match="head|available"):
        engine.answer("facts", {"q": {"type": "choice", "instructions": "select", "criteria": {"a": None, "b": None}}})
    assert engine.backbone.seen == []


@pytest.mark.parametrize("kwargs,state,question", [
    ({"max_ctx": 3}, "long evidence", {"type": "noul", "instructions": "verify facts"}),
    ({"max_opt": 2}, "x", {"type": "choice", "instructions": "select", "criteria": {"long option": None, "other": None}}),
    ({"backbone": TinyBackbone(seq_len=12)}, "facts", {"type": "noul", "instructions": "check the facts"}),
    ({"backbone": TinyBackbone(vocab=16)}, "facts", {"type": "noul", "instructions": "verify facts"}),
])
def test_token_overflow_and_small_vocab_rejected_without_forward(kwargs, state, question):
    engine = make_engine(**kwargs)
    with pytest.raises(ValueError):
        engine.answer(state, {"q": question})
    assert engine.backbone.seen == []


def test_indistinguishable_tokenized_options_rejected():
    engine = make_engine()
    engine.tokenizer = Mock(encode=lambda *args, **kwargs: [1])
    with pytest.raises(ValueError, match="identical|indistinguishable|duplicate"):
        engine.answer("facts", {"q": {"type": "choice", "instructions": "select", "criteria": {"a": None, "b": None}}})
    assert engine.backbone.seen == []


@pytest.mark.parametrize("settings", [
    {"temperature": 0}, {"temperature": -1}, {"temperature": math.nan},
    {"temperature": math.inf}, {"temperature": True}, {"mode": "bad"},
    {"layers": []}, {"layers": [1.5]}, {"layers": [True]},
    {"head_kind": "mlp", "layers": [-1, -2]},
    {"head_kind": "mlp", "layers": [-2]},
    {"head_kind": "attnpool", "mode": "marker"},
])
def test_invalid_head_settings(settings):
    from sciev.inference import HeadSettings

    with pytest.raises(ValueError):
        HeadSettings(**settings)


def test_serialized_access_to_shared_backbone():
    engine = make_engine()
    original = engine.backbone.forward
    state_lock = threading.Lock()
    active = 0
    peak = 0

    def forward(*args, **kwargs):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.01)
            return original(*args, **kwargs)
        finally:
            with state_lock:
                active -= 1

    engine.backbone.forward = forward
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: engine.answer("facts", {"q": {"type": "noul", "instructions": "verify facts"}}), range(4)))
    assert len(results) == 4
    assert peak == 1


@pytest.fixture
def bundle(tmp_path):
    from sciev.inference import HeadSettings

    torch.manual_seed(12)
    specs = {
        "choice": HeadSettings(head_kind="attnpool", layers=(-1, -2), canonical_order=True),
        "noul": HeadSettings(canonical_order=True),
        "score": HeadSettings(canonical_order=True),
    }
    manifest = {"release": "test", "backbone": "synthetic/backbone",
                "encoding": "systemone-v2", "seq_len": 2048, "heads": {}}
    checkpoints = {}
    for kind, settings in specs.items():
        head = AttnPoolHead(8, len(settings.layers)) if settings.head_kind == "attnpool" else DecisionHead(8)
        checkpoint = {"hf_backbone": manifest["backbone"], "head_kind": settings.head_kind,
                      "n_layers": len(settings.layers), "head": head.state_dict(),
                      "meta": {"mode": "r2_" + settings.mode}}
        path = tmp_path / (kind + ".pt")
        torch.save(checkpoint, path)
        manifest["heads"][kind] = {**asdict(settings), "file": path.name,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        checkpoints[kind] = checkpoint
    path = tmp_path / "bundle.json"

    def save():
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def replace_checkpoint(kind, checkpoint):
        head_path = tmp_path / manifest["heads"][kind]["file"]
        torch.save(checkpoint, head_path)
        manifest["heads"][kind]["sha256"] = hashlib.sha256(head_path.read_bytes()).hexdigest()
        save()

    save()
    return manifest, checkpoints, save, replace_checkpoint


def test_manifest_one_backbone_strict_heads_weights_only_cpu(bundle, monkeypatch):
    from sciev.inference import DecisionEngine

    manifest, checkpoints, save, _ = bundle
    factory = Mock(side_effect=lambda *args, **kwargs: TinyBackbone(seq_len=kwargs["seq_len"]))
    tokenizer_factory = Mock(return_value=FakeTokenizer())
    real_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    engine = DecisionEngine.from_manifest(save(), device="cpu", backbone_factory=factory,
                                          tokenizer_factory=tokenizer_factory)
    factory.assert_called_once()
    assert factory.call_args.args == (manifest["backbone"],)
    tokenizer_factory.assert_called_once()
    assert len(calls) == 3
    assert all(call["weights_only"] is True and call["map_location"] == "cpu" for call in calls)
    for kind, checkpoint in checkpoints.items():
        assert all(torch.equal(value, engine.heads[kind].state_dict()[key])
                   for key, value in checkpoint["head"].items())
    result = engine.answer("facts", all_questions())
    assert result["usage"]["forward_passes"] == 3
    info = engine.model_info()
    assert info["model"] == "sciev-test"
    assert info["available_types"] == ["choice", "noul", "score"]
    assert info["encoding"] == "systemone-v2"
    assert info["calibration"]["verified"] is False
    assert ".pt" not in json.dumps(info)
    assert str(save().parent) not in json.dumps(info)


def test_all_hashes_checked_before_deserialization_or_model_load(bundle, monkeypatch):
    from sciev.inference import DecisionEngine

    manifest, _, save, _ = bundle
    manifest["heads"]["score"]["sha256"] = "0" * 64
    factory = Mock(side_effect=AssertionError("must not load backbone"))
    tokenizer_factory = Mock(side_effect=AssertionError("must not load tokenizer"))
    deserialize = Mock(side_effect=AssertionError("must not deserialize"))
    monkeypatch.setattr(torch, "load", deserialize)
    with pytest.raises(ValueError, match="SHA|hash|checksum"):
        DecisionEngine.from_manifest(save(), backbone_factory=factory, tokenizer_factory=tokenizer_factory)
    factory.assert_not_called()
    tokenizer_factory.assert_not_called()
    deserialize.assert_not_called()


@pytest.mark.parametrize("change", [
    {"hf_backbone": "wrong/backbone"},
    {"lora_adapter": "unexpected-adapter"},
    {"head_kind": "attnpool"},
    {"n_layers": 2},
    {"meta": {"mode": "r2_marker"}},
    {"meta": {"mode": "r2_spanpool", "layers": [-2]}},
    {"meta": {"mode": "r2_spanpool", "canonical_order": False}},
    {"meta": {"mode": "r2_spanpool", "temperature": 2.0}},
    {"encoding": "old-encoding"},
])
def test_checkpoint_bundle_mismatches_rejected_before_backbone(bundle, change):
    from sciev.inference import DecisionEngine

    _, checkpoints, save, replace = bundle
    replace("noul", {**checkpoints["noul"], **change})
    factory = Mock(side_effect=AssertionError("must not load incompatible model"))
    with pytest.raises(ValueError):
        DecisionEngine.from_manifest(save(), backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


def test_missing_checkpoint_adapter_rejected_when_bundle_declares_one(bundle):
    from sciev.inference import DecisionEngine

    manifest, _, save, _ = bundle
    manifest["lora_adapter"] = "expected-adapter"
    factory = Mock(side_effect=AssertionError("must not load incompatible model"))
    with pytest.raises(ValueError, match="adapter|LoRA"):
        DecisionEngine.from_manifest(save(), backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


@pytest.mark.parametrize("corrupt", ["missing", "extra", "nonfinite"])
def test_head_state_is_strict_and_finite_before_backbone(bundle, corrupt):
    from sciev.inference import DecisionEngine

    _, checkpoints, save, replace = bundle
    checkpoint = checkpoints["score"]
    if corrupt == "missing":
        del checkpoint["head"]["net.0.weight"]
    elif corrupt == "extra":
        checkpoint["head"]["unexpected"] = torch.zeros(1)
    else:
        checkpoint["head"]["net.0.weight"][0] = float("nan")
    replace("score", checkpoint)
    factory = Mock(side_effect=AssertionError("must not load invalid state"))
    with pytest.raises(ValueError):
        DecisionEngine.from_manifest(save(), backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


@pytest.mark.parametrize("encoding", [None, "legacy-v1"])
def test_legacy_encoding_requires_opt_in_and_is_visibly_unverified(bundle, encoding):
    from sciev.inference import DecisionEngine

    manifest, _, save, _ = bundle
    if encoding is None:
        manifest.pop("encoding")
    else:
        manifest["encoding"] = encoding
    factory = Mock(side_effect=lambda *args, **kwargs: TinyBackbone())
    with pytest.raises(ValueError, match="encoding|legacy"):
        DecisionEngine.from_manifest(save(), backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()
    engine = DecisionEngine.from_manifest(save(), allow_legacy_encoding=True,
                                          backbone_factory=factory, tokenizer=FakeTokenizer())
    info = engine.model_info()
    assert info["bundle_encoding"] == encoding
    assert info["encoding"] == "systemone-v2"
    assert info["calibration"]["status"] == "legacy_unverified"
    assert info["calibration"]["verified"] is False
    result = engine.answer("facts", {"q": {"type": "noul", "instructions": "verify facts"}})
    assert result["metadata"]["calibration"] == info["calibration"]


def test_uniform_ties_choose_same_named_option_after_permutation():
    engine = make_engine()
    for parameter in engine.heads["choice"].parameters():
        parameter.zero_()
    question = {"type": "choice", "instructions": "select",
                "criteria": {"zebra": None, "ant": None, "yak": None}}
    first = engine.answer("facts", {"q": question})["answers"]["q"]
    question["criteria"] = dict(reversed(list(question["criteria"].items())))
    second = engine.answer("facts", {"q": question})["answers"]["q"]
    assert first["choice"] == second["choice"]
    assert first["probabilities"] == second["probabilities"]
    assert first["confidence"] == 0.0
    assert sum(first["probabilities"].values()) == pytest.approx(1.0, abs=1e-12)


def test_noul_is_yes_probability_not_winner_probability():
    engine = make_engine()
    engine.heads["noul"].forward = lambda h: torch.tensor([5.0, -5.0])
    answer = engine.answer("facts", {"q": {"type": "noul", "instructions": "verify facts"}})["answers"]["q"]
    assert answer["noul"] == pytest.approx(1 / (1 + math.exp(10 / 1.3)))
    assert answer["noul"] < 0.01
    assert answer["max_probability"] > 0.99


def test_original_name_mapping_with_known_presented_logits():
    engine = make_engine()
    engine.heads["choice"].forward = lambda h, bounds: torch.tensor([-3.0, 8.0, 1.0])
    question = {"type": "choice", "instructions": "select",
                "criteria": {"zebra": None, "ant": None, "yak": None}}
    answer = engine.answer("facts", {"q": question})["answers"]["q"]
    assert answer["choice"] == "yak"
    assert answer["probabilities"]["yak"] > answer["probabilities"]["zebra"] > answer["probabilities"]["ant"]


def test_marker_settings_invoke_real_head_and_count_marker_tokens():
    from sciev.decisions import encode_question

    engine = make_engine(heads={"choice": DecisionHead(8)},
                         settings={"choice": {"mode": "marker"}})
    question = {"type": "choice", "instructions": "select", "criteria": {"b": None, "a": None}}
    row = encode_question(engine.tokenizer, "facts", question)
    result = engine.answer("facts", {"q": question})
    assert len(engine.backbone.seen) == 1
    assert result["usage"]["input_tokens"] == len(row["ctx"]) + sum(map(len, row["opts"])) + 2
    assert result["usage"]["forward_passes"] == 1
    assert int((engine.backbone.seen[0] == engine.backbone.mask_id).sum()) == 2


def test_shared_adapter_paths_are_resolved_against_manifest_parent(bundle):
    from sciev.inference import DecisionEngine

    manifest, checkpoints, save, replace = bundle
    manifest["lora_adapter"] = "adapters/shared"
    for kind, checkpoint in checkpoints.items():
        replace(kind, {**checkpoint, "lora_adapter": "adapters/shared"})
    factory = Mock(side_effect=lambda *args, **kwargs: TinyBackbone())
    DecisionEngine.from_manifest(save(), device="cpu", backbone_factory=factory, tokenizer=FakeTokenizer())
    assert factory.call_args.kwargs["lora_adapter"] == str(save().parent / "adapters/shared")
    factory.assert_called_once()


@pytest.mark.parametrize("checksum", [None, "", "not-a-hash", "0" * 63, 42])
def test_supplied_checksums_must_be_valid_sha256(bundle, checksum):
    from sciev.inference import DecisionEngine

    manifest, _, save, _ = bundle
    manifest["heads"]["choice"]["sha256"] = checksum
    factory = Mock(side_effect=AssertionError("must validate checksum before model load"))
    with pytest.raises(ValueError, match="SHA|checksum"):
        DecisionEngine.from_manifest(save(), device="cpu", backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


def test_maximum_choice_count_keeps_normalized_full_precision_probabilities():
    engine = make_engine(heads={"choice": DecisionHead(8)}, settings={"choice": {}})
    for parameter in engine.heads["choice"].parameters():
        parameter.zero_()
    question = {"type": "choice", "instructions": "select",
                "criteria": {f"opt{i:03}": None for i in range(255)}}
    result = engine.answer("facts", {"q": question})
    probabilities = result["answers"]["q"]["probabilities"]
    assert len(probabilities) == 255
    assert math.fsum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(value == 1 / 255 for value in probabilities.values())


@pytest.mark.parametrize("settings", [
    {"head_kind": "attnpool", "layers": [-1]},
    {"head_kind": "attnpool", "layers": [-1, -3]},
    {"head_kind": "mlp", "layers": [-1]},
])
def test_injected_head_class_and_actual_layer_count_must_match(settings):
    with pytest.raises(ValueError, match="head|layer"):
        make_engine(heads={"choice": AttnPoolHead(8, n_layers=2)}, settings={"choice": settings})


@pytest.mark.parametrize("kwargs", [{"max_ctx": 0}, {"max_opt": 0}, {"max_questions": 0}])
def test_engine_token_and_question_budgets_must_be_positive(kwargs):
    with pytest.raises(ValueError, match="positive"):
        make_engine(**kwargs)


def test_nonfinite_head_output_is_model_failure_not_probabilities():
    engine = make_engine()
    engine.heads["noul"].forward = lambda h: torch.full((h.shape[0],), float("nan"))
    with pytest.raises(RuntimeError, match="finite|logit|probabilit"):
        engine.answer("facts", {"q": {"type": "noul", "instructions": "verify facts"}})


@pytest.mark.parametrize("encoding", [None, "legacy-v1"])
@pytest.mark.parametrize("explicit_inference", [False, True])
def test_released_legacy_mlp_requested_layers_need_explicit_opt_in(bundle, encoding, explicit_inference):
    from sciev.inference import DecisionEngine

    manifest, checkpoints, save, replace = bundle
    if encoding is None:
        manifest.pop("encoding")
    else:
        manifest["encoding"] = encoding
    for kind in ("noul", "score"):
        checkpoint = {**checkpoints[kind], "n_layers": 4}
        if explicit_inference:
            checkpoint["inference"] = {
                "head_kind": "mlp", "n_layers": 1, "mode": "spanpool",
                "layers": [-1], "canonical_order": True,
            }
        replace(kind, checkpoint)
    factory = Mock(side_effect=lambda *args, **kwargs: TinyBackbone())
    with pytest.raises(ValueError, match="encoding|legacy"):
        DecisionEngine.from_manifest(save(), device="cpu", backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()
    engine = DecisionEngine.from_manifest(
        save(), device="cpu", allow_legacy_encoding=True,
        backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_called_once()
    for kind in ("noul", "score"):
        assert isinstance(engine.heads[kind], DecisionHead)
        assert engine.settings[kind].layers == (-1,)
        assert all(torch.equal(value, engine.heads[kind].state_dict()[key])
                   for key, value in checkpoints[kind]["head"].items())
    result = engine.answer("facts", all_questions())
    assert result["usage"]["forward_passes"] == 3
    assert result["metadata"]["calibration"]["status"] == "legacy_unverified"
    assert result["metadata"]["calibration"]["verified"] is False
    assert engine.model_info()["calibration"] == result["metadata"]["calibration"]


@pytest.mark.parametrize("kind,encoding", [
    ("choice", None), ("choice", "legacy-v1"),
    ("noul", "systemone-v2"), ("score", "systemone-v2"),
])
def test_legacy_opt_in_does_not_relax_attnpool_or_modern_layer_counts(bundle, kind, encoding):
    from sciev.inference import DecisionEngine

    manifest, checkpoints, save, replace = bundle
    if encoding is None:
        manifest.pop("encoding")
    else:
        manifest["encoding"] = encoding
    replace(kind, {**checkpoints[kind], "n_layers": 4})
    factory = Mock(side_effect=AssertionError("must reject incompatible layer counts"))
    with pytest.raises(ValueError, match="layer count"):
        DecisionEngine.from_manifest(
            save(), device="cpu", allow_legacy_encoding=True,
            backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


@pytest.mark.parametrize("change", [
    {"inference": {"n_layers": 4}},
    {"inference": {"layers": [-1, -2]}},
    {"inference": {"layers": [-2]}},
    {"inference": {"mode": "marker"}},
    {"inference": {"canonical_order": False}},
    {"inference": {"temperature": 2.0}},
    {"meta": {"mode": "r2_spanpool", "n_layers": 4}},
    {"layers": [-1, -2]},
])
def test_legacy_mlp_compatibility_does_not_ignore_explicit_inference_settings(bundle, change):
    from sciev.inference import DecisionEngine

    manifest, checkpoints, save, replace = bundle
    manifest.pop("encoding")
    replace("noul", {**checkpoints["noul"], "n_layers": 4, **change})
    factory = Mock(side_effect=AssertionError("must reject contradictory inference settings"))
    with pytest.raises(ValueError):
        DecisionEngine.from_manifest(
            save(), device="cpu", allow_legacy_encoding=True,
            backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


@pytest.mark.parametrize("count", [0, True, "4"])
def test_legacy_mlp_requested_layer_count_must_still_be_a_positive_integer(bundle, count):
    from sciev.inference import DecisionEngine

    manifest, checkpoints, save, replace = bundle
    manifest.pop("encoding")
    replace("noul", {**checkpoints["noul"], "n_layers": count})
    factory = Mock(side_effect=AssertionError("must reject malformed legacy metadata"))
    with pytest.raises(ValueError, match="positive integer"):
        DecisionEngine.from_manifest(
            save(), device="cpu", allow_legacy_encoding=True,
            backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


def test_legacy_mlp_compatibility_still_strictly_validates_actual_head_state(bundle):
    from sciev.inference import DecisionEngine

    manifest, checkpoints, save, replace = bundle
    manifest.pop("encoding")
    checkpoint = {**checkpoints["noul"], "n_layers": 4}
    del checkpoint["head"]["net.0.weight"]
    replace("noul", checkpoint)
    factory = Mock(side_effect=AssertionError("must validate actual MLP state before loading backbone"))
    with pytest.raises(ValueError, match="strictly match"):
        DecisionEngine.from_manifest(
            save(), device="cpu", allow_legacy_encoding=True,
            backbone_factory=factory, tokenizer=FakeTokenizer())
    factory.assert_not_called()


def test_r2_uses_shared_probabilities_for_every_specialist_head(monkeypatch):
    from sciev import decisions

    shared = Mock(wraps=decisions.decision_probabilities)
    monkeypatch.setattr(decisions, "decision_probabilities", shared)
    engine = make_engine()
    engine.answer("facts", all_questions())
    assert shared.call_count == 3
    assert [call.args[1] for call in shared.call_args_list] == [0.8, 1.3, 0.9]
