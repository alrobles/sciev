import importlib
import json
import sys
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from test_inference import FakeTokenizer, all_questions, make_engine


@pytest.mark.parametrize("configured", [False, True])
def test_import_never_loads_models_or_tokenizers(monkeypatch, configured):
    from reverse_jev import inference, model
    from transformers import AutoTokenizer

    loader = Mock(side_effect=AssertionError("import must not load a model"))
    monkeypatch.setattr(model, "load_backbone", loader)
    monkeypatch.setattr(model, "HFBackbone", loader)
    monkeypatch.setattr(inference, "HFBackbone", loader)
    monkeypatch.setattr(torch, "load", loader)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loader)
    for key in ("SCIEV_MANIFEST", "REVJEV_CKPT"):
        if configured:
            monkeypatch.setenv(key, "/private/not-a-real-checkpoint")
        else:
            monkeypatch.delenv(key, raising=False)
    if "reverse_jev.serve" in sys.modules:
        module = importlib.reload(sys.modules["reverse_jev.serve"])
    else:
        module = importlib.import_module("reverse_jev.serve")
    assert module.app is not None
    assert callable(module.create_app)
    loader.assert_not_called()


def test_injected_api_all_types_metadata_and_usage(monkeypatch):
    from reverse_jev.serve import create_app
    from reverse_jev import readout

    monkeypatch.setenv("SCIEV_MANIFEST", "/private/invalid-manifest.json")
    monkeypatch.setattr(readout, "answer_questions", Mock(side_effect=AssertionError("R2 must not use R1")))
    engine = make_engine()
    with TestClient(create_app(engine)) as client:
        info = client.get("/v1/models")
        assert info.status_code == 200
        assert info.json()["readout"] == "r2"
        assert info.json()["available_types"] == ["choice", "noul", "score"]
        assert info.json()["limits"]["choice"]["max_options"] == 255
        assert info.json()["limits"]["score"]["max_levels"] == 10
        result = client.post("/v1/systemone", json={
            "state": {"observations": ["a", "b"]}, "model": "synthetic-r2",
            "questions": all_questions(),
        })
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["model"] == "synthetic-r2"
    assert set(body["answers"]) == {"select", "verify", "rate"}
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["forward_passes"] == 3
    assert body["usage"]["input_tokens"] == sum(ids.numel() for ids in engine.backbone.seen)
    assert body["latency_ms"] >= 0
    assert body["metadata"]["encoding"] == "systemone-v2"
    assert body["metadata"]["calibration"]["verified"] is False
    assert "concentration" in body["metadata"]["confidence_semantics"]
    assert "/private" not in info.text


def test_ids_and_labels_are_ignored_by_api_features():
    from reverse_jev.serve import create_app

    engine = make_engine()
    question = {"type": "choice", "instructions": ["select"],
                "criteria": {"one": {"description": "first"}, "two": ["second"]}, "label": 0}
    with TestClient(create_app(engine)) as client:
        first = client.post("/v1/systemone", json={"state": ["facts"], "questions": {"original-id": question}})
        question["label"] = 1
        question["id"] = "not-a-feature"
        second = client.post("/v1/systemone", json={"state": ["facts"], "questions": {"changed-id": question}})
    assert first.status_code == second.status_code == 200
    assert first.json()["answers"]["original-id"] == second.json()["answers"]["changed-id"]
    assert torch.equal(engine.backbone.seen[0], engine.backbone.seen[1])


@pytest.mark.parametrize("payload", [
    {"state": 42, "questions": {"q": {"type": "noul", "instructions": "verify facts"}}},
    {"state": None, "questions": {"q": {"type": "noul", "instructions": "verify facts"}}},
    {"state": "facts", "questions": {}},
    {"state": "facts", "questions": {"q": {"type": "unknown"}}},
    {"state": "facts", "questions": {"q": {"type": "choice", "instructions": "select", "criteria": {"a": None}}}},
    {"state": "facts", "questions": {"q": {"type": "choice", "instructions": "select", "criteria": {str(i): None for i in range(256)}}}},
    {"state": "facts", "questions": {"q": {"type": "score", "criteria": ["a"]}}},
    {"state": "facts", "questions": {"q": {"type": "score", "criteria": list(range(11))}}},
    {"state": "facts", "questions": {"q": {"type": "noul", "criteria": {"true": "ok"}}}},
    {"state": "facts", "questions": {"q": {"type": "noul", "criteria": {"true": "ok", "false": "no", "other": None}}}},
    {"state": "facts", "questions": {"q": {"type": "choice", "instructions": "select", "criteria": ["a", "b"]}}},
    {"state": "facts", "questions": {"q": {"type": "noul", "instructions": 17}}},
    {"state": "facts", "questions": {"q": {"type": "choice", "instructions": "select", "criteria": {"": None, "a": None}}}},
    {"state": "facts", "questions": {"valid": {"type": "noul", "instructions": "verify facts"}, "invalid": {"type": "score", "criteria": []}}},
])
def test_public_schema_rejects_invalid_inputs_without_inference(payload):
    from reverse_jev.serve import create_app

    engine = make_engine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422, response.text
    assert engine.backbone.seen == []


def test_oversize_encoder_input_is_clear_4xx_not_truncation():
    from reverse_jev.serve import create_app

    engine = make_engine(max_ctx=12)
    with TestClient(create_app(engine)) as client:
        result = client.post("/v1/systemone", json={
            "state": "far too much context for this token budget", "questions": {"q": {"type": "noul", "instructions": "verify facts"}},
        })
    assert result.status_code == 422
    assert engine.backbone.seen == []


def test_unknown_model_and_missing_head_are_clear_4xx():
    from reverse_jev.serve import create_app
    from reverse_jev.model import DecisionHead

    engine = make_engine(heads={"noul": DecisionHead(8)}, settings={"noul": {}})
    with TestClient(create_app(engine)) as client:
        unknown = client.post("/v1/systemone", json={
            "state": "facts", "model": "not-installed", "questions": {"q": {"type": "noul", "instructions": "verify facts"}},
        })
        missing = client.post("/v1/systemone", json={
            "state": "facts", "questions": {"q": {"type": "choice", "instructions": "select", "criteria": {"a": None, "b": None}}},
        })
    assert unknown.status_code == 404
    assert missing.status_code == 422
    assert engine.backbone.seen == []


def test_no_configuration_is_503_without_load_and_warns_local_only(monkeypatch, caplog):
    from reverse_jev.serve import create_app

    monkeypatch.delenv("SCIEV_MANIFEST", raising=False)
    monkeypatch.delenv("REVJEV_CKPT", raising=False)
    with TestClient(create_app()) as client:
        response = client.get("/v1/models")
    assert response.status_code == 503
    assert "auth" in caplog.text.lower()
    assert "localhost" in caplog.text.lower()


def test_bad_r2_configuration_does_not_fall_back_to_legacy(monkeypatch):
    from reverse_jev.serve import create_app
    from reverse_jev.inference import DecisionEngine
    from reverse_jev import model

    monkeypatch.setenv("SCIEV_MANIFEST", "/private/broken-bundle.json")
    monkeypatch.setenv("REVJEV_CKPT", "/private/legacy-checkpoint.pt")
    r2 = Mock(side_effect=ValueError("private configuration mismatch"))
    legacy = Mock(side_effect=AssertionError("must not fall back to R1"))
    monkeypatch.setattr(DecisionEngine, "from_manifest", r2)
    monkeypatch.setattr(model, "load_backbone", legacy)
    api = create_app()
    r2.assert_not_called()
    with TestClient(api) as client:
        response = client.get("/v1/models")
    assert response.status_code == 503
    assert "private" not in response.text
    r2.assert_called_once()
    legacy.assert_not_called()


def test_lazy_manifest_load_once_and_explicit_encoding_env(monkeypatch):
    from reverse_jev.serve import create_app
    from reverse_jev.inference import DecisionEngine

    engine = make_engine()
    loader = Mock(return_value=engine)
    monkeypatch.setenv("SCIEV_MANIFEST", "synthetic.json")
    monkeypatch.setenv("SCIEV_ALLOW_LEGACY_ENCODING", "1")
    monkeypatch.setattr(DecisionEngine, "from_manifest", loader)
    api = create_app()
    loader.assert_not_called()
    with TestClient(api) as client:
        assert client.get("/v1/models").status_code == 200
        assert client.get("/v1/models").status_code == 200
        assert client.post("/v1/systemone", json={"state": "facts", "questions": {"q": {"type": "noul", "instructions": "verify facts"}}}).status_code == 200
    loader.assert_called_once()
    assert loader.call_args.kwargs["allow_legacy_encoding"] is True


def test_model_failures_are_not_returned_as_successful_probabilities():
    from reverse_jev.serve import create_app

    engine = make_engine()
    engine.backbone.forward = Mock(side_effect=RuntimeError("failed device kernel"))
    with TestClient(create_app(engine), raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", json={"state": "facts", "questions": {"q": {"type": "noul", "instructions": "verify facts"}}})
    assert response.status_code == 500
    assert "probabilities" not in response.text


def test_openapi_has_discriminated_public_question_schema():
    from reverse_jev.serve import create_app

    schema = create_app(make_engine()).openapi()
    rendered = json.dumps(schema)
    assert "ChoiceQuestion" in rendered
    assert "NoulQuestion" in rendered
    assert "ScoreQuestion" in rendered
    assert '"discriminator"' in rendered


@pytest.mark.parametrize("question_type,criteria", [
    ("choice", {"a": 17, "b": None}),
    ("score", ["low", 17]),
    ("noul", {"true": "supported", "false": 17}),
])
def test_public_schema_rejects_scalar_descriptors(question_type, criteria):
    from pydantic import ValidationError
    from reverse_jev.serve import SystemOneRequest

    with pytest.raises(ValidationError):
        SystemOneRequest(state="facts", questions={
            "q": {"type": question_type, "instructions": "judge", "criteria": criteria},
        })


def test_legacy_encoding_status_is_visible_on_both_endpoints():
    from reverse_jev.serve import create_app

    engine = make_engine(bundle_encoding=None, allow_legacy_encoding=True)
    with TestClient(create_app(engine)) as client:
        info = client.get("/v1/models")
        result = client.post("/v1/systemone", json={
            "state": "facts", "questions": {"q": {"type": "noul", "instructions": "verify facts"}},
        })
    assert info.status_code == result.status_code == 200
    assert info.json()["calibration"]["status"] == "legacy_unverified"
    assert result.json()["metadata"]["calibration"]["status"] == "legacy_unverified"
    assert result.json()["metadata"]["bundle_encoding"] is None


def test_explicit_r1_legacy_path_is_marked_and_usage_is_real(monkeypatch, caplog):
    from reverse_jev.serve import create_app
    from reverse_jev import model
    from transformers import AutoTokenizer
    from test_inference import TinyBackbone

    backbone = TinyBackbone()
    monkeypatch.delenv("SCIEV_MANIFEST", raising=False)
    monkeypatch.setenv("REVJEV_CKPT", "/private/native-checkpoint.pt")
    monkeypatch.setattr(model, "load_backbone", Mock(return_value=backbone))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=FakeTokenizer()))
    with TestClient(create_app()) as client:
        info = client.get("/v1/models")
        result = client.post("/v1/systemone", json={"state": "facts", "questions": all_questions()})
    assert info.status_code == result.status_code == 200, result.text
    metadata = info.json()
    assert metadata["readout"] == "r1"
    assert metadata["legacy"] is True
    assert metadata["calibration"]["status"] == "legacy_unverified"
    assert "private" not in info.text
    assert "checkpoint.pt" not in info.text
    assert "legacy" in caplog.text.lower()
    assert result.json()["usage"] == {
        "input_tokens": sum(ids.numel() for ids in backbone.seen),
        "output_tokens": 0, "forward_passes": 3,
    }
    assert all(backbone.inference_flags)


@pytest.mark.parametrize("uniform", [False, True])
def test_api_exactly_matches_shared_posterior_and_token_order_selection(monkeypatch, uniform):
    from reverse_jev import decisions
    from reverse_jev.serve import create_app

    class ReverseTokenOrder:
        def encode(self, text, add_special_tokens=False):
            return [256 - byte for byte in text.encode("utf-8")]

    engine = make_engine()
    engine.tokenizer = ReverseTokenOrder()
    if uniform:
        for parameter in engine.heads["choice"].parameters():
            parameter.zero_()
    normalize = decisions.decision_probabilities
    predict = decisions.decision_prediction
    normalize_spy = Mock(wraps=normalize)
    predict_spy = Mock(wraps=predict)
    monkeypatch.setattr(decisions, "decision_probabilities", normalize_spy)
    monkeypatch.setattr(decisions, "decision_prediction", predict_spy)
    question = {"type": "choice", "instructions": "select",
                "criteria": {"ant": None, "yak": None, "zebra": None}}
    settings = engine.settings["choice"]
    answers = []
    with TestClient(create_app(engine)) as client:
        for _ in range(2):
            row = decisions.encode_question(engine.tokenizer, "facts", question)
            with torch.inference_mode():
                logits, prepared = decisions.decision_logits(
                    engine.backbone, engine.heads["choice"], row["ctx"], row["opts"], "cpu",
                    mode=settings.mode, layers=settings.layers, canonical=settings.canonical_order,
                    strict=True)
                probabilities = normalize(logits, settings.temperature)
                selected_index = predict(logits, prepared)
            expected_choice = row["option_keys"][selected_index]
            if uniform:
                assert expected_choice == "zebra"
                assert expected_choice != min(question["criteria"])
            response = client.post("/v1/systemone", json={
                "state": "facts", "questions": {"q": question},
            })
            assert response.status_code == 200, response.text
            answer = response.json()["answers"]["q"]
            assert answer["choice"] == expected_choice
            assert answer["probabilities"] == dict(zip(row["option_keys"], probabilities))
            assert answer["max_probability"] == max(probabilities)
            assert response.json()["usage"]["forward_passes"] == 1
            answers.append(answer)
            question["criteria"] = dict(reversed(list(question["criteria"].items())))
    assert answers[0] == answers[1]
    assert normalize_spy.call_count == 2
    assert predict_spy.call_count == 2
