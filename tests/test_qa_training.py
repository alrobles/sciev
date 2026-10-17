import json
from types import SimpleNamespace

import pytest
import torch

from sciev import qa_train
from sciev.model import MdLMMoE


def row(index, source="arc_easy"):
    return {"qid": f"q{index}", "ctx": [index + 1], "opts": [[21], [22], [23], [24]],
            "gold": index % 4, "kind": "choice", "encoding": "systemone-v2",
            "schema_version": 2, "split_group": f"group{index}", "source": source}


def test_qa_training_explicitly_forbids_gpqa_labels():
    with pytest.raises(ValueError, match="GPQA"):
        qa_train.require_training_data([row(1, "gpqa")])


def test_selection_subset_preserves_groups_and_is_deterministic():
    rows = [row(i) for i in range(6)]
    rows[1]["split_group"] = rows[0]["split_group"]
    selected = qa_train.selection_subset(rows, 3, 7331)
    assert selected == qa_train.selection_subset(rows, 3, 7331)
    assert (rows[0] in selected) == (rows[1] in selected)


def test_restore_trainable_never_changes_frozen_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    state = qa_train.trainable_state(model)
    assert set(state) == {"1.weight", "1.bias"}
    qa_train.restore_trainable(model, {name: value + 1 for name, value in state.items()})
    assert torch.equal(model[0].weight, original["0.weight"])
    assert torch.equal(model[1].weight, original["1.weight"] + 1)
    with pytest.raises(ValueError, match="parameters"):
        qa_train.restore_trainable(model, {})


def test_qa_evaluation_fits_disjoint_temperature_and_policy_sets(tmp_path, monkeypatch):
    from sciev import qa_eval
    from sciev.model import AttnPoolHead
    from sciev.train import r2_checkpoint

    model = MdLMMoE(vocab=32, hidden=8, layers=1, heads=2, ff_mult=2,
                    seq_len=32, n_experts=1, k=1)
    head = AttnPoolHead(8)
    args = SimpleNamespace(head_kind="attnpool", layers_list=[-1], r2_mode="spanpool",
                           canonical_order=True, strict_inputs=True, decision_type="choice",
                           pairs_train=None, decisions_train=None, training_contract="test", seed=1)
    checkpoint = r2_checkpoint(model, head, [row(i) for i in range(4)], args)
    checkpoint.update(model=model.state_dict(), model_config=model.model_config)
    ckpt = tmp_path / "decision.pt"
    torch.save(checkpoint, ckpt)
    for name, indices in (("calibration", range(4, 7)), ("policy", range(7, 10)), ("eval", range(10, 13))):
        (tmp_path / f"qa_choice_{name}.jsonl").write_text("\n".join(json.dumps(row(i)) for i in indices))
    out = tmp_path / "evaluation"
    monkeypatch.setattr("sys.argv", ["qa_eval", "--ckpt", str(ckpt), "--qa-dir", str(tmp_path),
                                     "--evaluation", f"control={tmp_path / 'qa_choice_eval.jsonl'}",
                                     "--out-dir", str(out), "--device", "cpu"])
    qa_eval.main()
    result = json.loads((out / "control.json").read_text())
    assert result["training_overlap"]["status"] == "verified"
    assert result["temperature_file"]["sha256"] != result["policy_file"]["sha256"]
    assert result["decisions_eval"]["n"] == 3
    assert result["decisions_eval"]["flip_rate"] == 0
    assert result["decisions_eval"]["selective"]["coverage"] == 0
    assert len(result["decisions_eval"]["predictions"]) == 3


def test_qa_driver_saves_and_resumes_a_budget_limited_run(tmp_path, monkeypatch):
    class TinyBackbone(MdLMMoE):
        def __init__(self, name, device):
            super().__init__(vocab=32, hidden=8, layers=1, heads=2, ff_mult=2,
                             seq_len=32, n_experts=1, k=1)
            self.hf_name = name
            self.hf = SimpleNamespace(config=SimpleNamespace(_commit_hash="test_revision"))
            self.to(device)

    monkeypatch.setattr(qa_train, "HFBackbone", TinyBackbone)
    train = tmp_path / "train.jsonl"
    select = tmp_path / "select.jsonl"
    train.write_text("\n".join(json.dumps(row(i)) for i in range(4)))
    select.write_text("\n".join(json.dumps(row(i)) for i in range(4, 6)))
    out = tmp_path / "run"
    argv = ["qa_train", "--decisions-train", str(train), "--selection-dev", str(select),
            "--out", str(out), "--device", "cpu", "--r2-layers=-1", "--steps", "4",
            "--accum", "2", "--save-every", "2", "--eval-every", "2", "--select-limit", "2"]
    monkeypatch.setattr("sys.argv", [*argv, "--max-seconds", "0.000001"])
    with pytest.raises(SystemExit) as error:
        qa_train.main()
    assert error.value.code == 75
    partial = json.loads((out / "progress.json").read_text())
    assert partial["completed_steps"] == 2 and not partial["budget_complete"]
    monkeypatch.setattr("sys.argv", [*argv, "--resume", str(out / "last_training.pt")])
    qa_train.main()
    assert json.loads((out / "progress.json").read_text())["budget_complete"]
    checkpoint = torch.load(out / "decision.pt", weights_only=True)
    assert checkpoint["inference"]["decision_type"] == "choice"
    assert checkpoint["selected_step"] in (2, 4)
    assert checkpoint["training_data"]["n_rows"] == 4
