from types import SimpleNamespace

import pytest
import torch

from scripts.eval_llada_pll import label_scores, option_pll
from scripts.eval_llada_prompted import diffuse_fill, parse_answer


class PositionModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids):
        self.calls.append(input_ids.clone())
        logits = torch.full((*input_ids.shape, 12), -20.0)
        for position in range(input_ids.shape[1]):
            logits[:, position, position + 1] = 20.0
        return SimpleNamespace(logits=logits)


def test_masked_option_scoring_uses_same_position_not_ar_shift():
    model = PositionModel()
    total, mean = option_pll(model, torch.tensor([[1, 2]]), [3, 4], 11)
    assert total == pytest.approx(0.0, abs=1e-6)
    assert mean == pytest.approx(0.0, abs=1e-6)
    assert model.calls[0].tolist() == [[1, 2, 11, 11]]


def test_true_pll_masks_exactly_one_answer_token_per_replica():
    model = PositionModel()
    total, mean = option_pll(model, torch.tensor([[1, 2]]), [3, 4], 11,
                             mode="leave_one_out", batch_size=2)
    assert total == mean == pytest.approx(0.0, abs=1e-6)
    assert model.calls[0].tolist() == [[1, 2, 11, 4], [1, 2, 3, 11]]


def test_label_scoring_uses_one_forward_for_all_choices():
    model = PositionModel()
    scores = label_scores(model, torch.tensor([[1, 2]]), [[3], [4], [5], [6]], 11)
    assert max(range(4), key=lambda i: scores[i][0]) == 0
    assert len(model.calls) == 1
    with pytest.raises(ValueError, match="single-token"):
        label_scores(model, torch.tensor([[1, 2]]), [[3, 4], [5]], 11)


@pytest.mark.parametrize("text,expected", [
    ("A", 0), ("(B)", 1), ("D.", 3), ("Final answer: C", 2),
    ("Analysis: A is incorrect.\nFinal answer: B", 1),
    ("Analysis of chemical bonds", -1), ("The answer is a chemical compound", -1),
    ("Final answer: H", -1), ("", -1),
])
def test_answer_parser_does_not_find_letters_inside_prose(text, expected):
    assert parse_answer(text, 4) == expected


def test_instruction_variants_do_not_swallow_schema_errors_or_remap_gold():
    from data.build_instr_variants import build_variant

    class Tokenizer:
        def encode(self, value, add_special_tokens=False):
            return list(value.encode())

    record = {"state": "Question: test", "sample_id": "sample", "split_group": "group",
              "questions": {"q": {"type": "choice", "instructions": "Choose.",
                                  "criteria": {"first": None, "second": None},
                                  "option_keys": ["second", "first"], "label": "second"}}}
    rows, _, exclusions = build_variant([record], Tokenizer(), "Select.", 1)
    assert rows[0]["gold"] == 1 and not exclusions
    record["questions"]["q"]["criteria"] = {"one": None}
    with pytest.raises(ValueError):
        build_variant([record], Tokenizer(), "Select.", 1, exclude_overflow=True)


def test_diffusion_schedule_completes_long_suffix():
    model = PositionModel()
    result = diffuse_fill(model, torch.tensor([[1]]), 11, 8, 4, "cpu")
    assert result.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    assert len(model.calls) == 4


@pytest.mark.parametrize("gen_len,steps", [(0, 1), (4, 0), (4, 5)])
def test_diffusion_rejects_invalid_step_budgets(gen_len, steps):
    with pytest.raises(ValueError):
        diffuse_fill(PositionModel(), torch.tensor([[1]]), 11, gen_len, steps, "cpu")


def test_diffusion_never_remasks_or_changes_the_prompt():
    class MaskPreferringModel:
        def __call__(self, input_ids):
            logits = torch.zeros((*input_ids.shape, 12))
            logits[..., 11] = 20
            logits[..., 3] = 10
            return SimpleNamespace(logits=logits)

    result = diffuse_fill(MaskPreferringModel(), torch.tensor([[11, 2]]), 11, 4, 2, "cpu")
    assert result.tolist() == [[11, 2, 3, 3, 3, 3]]
