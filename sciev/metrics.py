import math
from collections.abc import Mapping
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np


__all__ = [
    "classification_metrics",
    "fit_acceptance_policy",
    "evaluate_acceptance_policy",
    "oracle_automation_rate",
]

_POLICY_METHOD = "empirical_dev_threshold"
_POLICY_VERSION = 1
_POLICY_SCORE = "max_probability"


def _items(values, name):
    if isinstance(values, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(f"{name} must be an ordered sequence")
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an ordered sequence") from exc


def _unit_scalar(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite real number in [0, 1]")
    return float(value)


def _unit_vector(values, name):
    return np.asarray([_unit_scalar(value, name) for value in _items(values, name)],
                      dtype=float)


def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _confidence_correctness(confs, corrects, allow_empty=False):
    confs = _unit_vector(confs, "confs")
    corrects = _items(corrects, "corrects")
    if len(confs) != len(corrects):
        raise ValueError("confs and corrects must have aligned lengths")
    if not allow_empty and not len(confs):
        raise ValueError("confs and corrects must be nonempty")
    if any(not isinstance(value, (Real, np.bool_)) or value not in (0, 1)
           for value in corrects):
        raise ValueError("corrects must contain only finite binary values (0 or 1)")
    return confs, np.asarray(corrects, dtype=np.int64)


def _select_threshold(confs, corrects, error_budget, min_accepted):
    if min_accepted > len(confs):
        return None, 0, 0
    order = np.argsort(-confs, kind="stable")
    sorted_confs = confs[order]
    counts = np.arange(1, len(confs) + 1)
    errors = np.cumsum(1 - corrects[order])
    ends_of_ties = np.r_[sorted_confs[:-1] != sorted_confs[1:], True]
    feasible = np.flatnonzero(
        ends_of_ties & (counts >= min_accepted) & (errors / counts <= error_budget))
    if not len(feasible):
        return None, 0, 0
    last = feasible[-1]
    return float(sorted_confs[last]), int(counts[last]), int(errors[last])


def oracle_automation_rate(confs, corrects, error_budget=0.05):
    error_budget = _unit_scalar(error_budget, "error_budget")
    confs, corrects = _confidence_correctness(confs, corrects)
    _, accepted, _ = _select_threshold(confs, corrects, error_budget, 1)
    return accepted / len(confs)


def _ece(confs, corrects):
    edges = np.linspace(0, 1, 11)
    total = 0.0
    for index in range(10):
        above = confs >= edges[index] if index == 0 else confs > edges[index]
        selected = above & (confs <= edges[index + 1])
        if selected.any():
            total += float(selected.mean()) * abs(
                float(confs[selected].mean()) - float(corrects[selected].mean()))
    return total


def classification_metrics(prob_rows, golds, *, ordinal=False, fixed_labels=False,
                           predictions=None):
    if not isinstance(ordinal, (bool, np.bool_)) or not isinstance(
            fixed_labels, (bool, np.bool_)):
        raise ValueError("ordinal and fixed_labels must be booleans")
    if ordinal and not fixed_labels:
        raise ValueError("ordinal metrics require fixed_labels=True with ordered labels")
    rows = _items(prob_rows, "prob_rows")
    golds = _items(golds, "golds")
    if not rows or len(rows) != len(golds):
        raise ValueError("prob_rows and golds must be nonempty with aligned lengths")
    validated_rows, validated_golds = [], []
    for index, (row, gold) in enumerate(zip(rows, golds)):
        row = _unit_vector(row, f"prob_rows[{index}]")
        if len(row) < 2:
            raise ValueError(f"prob_rows[{index}] must have at least two probabilities")
        if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"prob_rows[{index}] probabilities must sum to one")
        gold = _integer(gold, f"golds[{index}]", 0)
        if gold >= len(row):
            raise ValueError(f"golds[{index}] is outside its probability row")
        validated_rows.append(row)
        validated_golds.append(gold)
    rows = validated_rows
    golds = np.asarray(validated_golds, dtype=np.int64)
    n = len(rows)
    n_labels = len(rows[0])
    if fixed_labels and any(len(row) != n_labels for row in rows):
        raise ValueError("fixed_labels requires the same label count in every row")
    if predictions is None:
        predictions = [int(row.argmax()) for row in rows]
    else:
        predictions = _items(predictions, "predictions")
        if len(predictions) != n:
            raise ValueError("predictions must have the same length as prob_rows and golds")
        for index, (row, prediction) in enumerate(zip(rows, predictions)):
            prediction = _integer(prediction, f"predictions[{index}]", 0)
            if prediction >= len(row):
                raise ValueError(f"predictions[{index}] is outside its probability row")
            if row[prediction] != row.max():
                raise ValueError(f"predictions[{index}] must select a maximum-probability class")
            predictions[index] = prediction
    predictions = np.asarray(predictions, dtype=np.int64)
    confs = np.asarray([float(row.max()) for row in rows])
    corrects = (predictions == golds).astype(np.int64)
    gold_probs = [float(row[gold]) for row, gold in zip(rows, golds)]
    zero_gold_count = sum(prob == 0 for prob in gold_probs)
    briers = []
    for row, gold in zip(rows, golds):
        residual = row.copy()
        residual[gold] -= 1
        briers.append(float(residual @ residual))
    result = {
        "n": n,
        "acc": float(corrects.mean()),
        "nll": None if zero_gold_count else math.fsum(-math.log(p) for p in gold_probs) / n,
        "nll_zero_gold_count": zero_gold_count,
        "brier": math.fsum(briers) / n,
        "ece": _ece(confs, corrects),
        "oracle_automation_5pct": oracle_automation_rate(confs, corrects),
    }
    if fixed_labels:
        support = np.bincount(golds, minlength=n_labels)
        predicted_support = np.bincount(predictions, minlength=n_labels)
        true_positives = np.bincount(golds[corrects.astype(bool)], minlength=n_labels)
        present = support > 0
        denominators = support + predicted_support
        f1 = np.divide(2 * true_positives, denominators,
                       out=np.zeros(n_labels, dtype=float), where=denominators > 0)
        result.update({
            "class_support": support.tolist(),
            "majority_baseline_acc": int(support.max()) / n,
            "majority_baseline_note": (
                "Largest observed true class fraction; not a constant classifier selected on dev."),
            "balanced_acc": float((true_positives[present] / support[present]).mean()),
            "balanced_acc_label_scope": "positive_true_support",
            "macro_f1": float(f1.mean()),
            "macro_f1_label_scope": "all_fixed_labels",
            "macro_f1_zero_division": 0.0,
        })
    if ordinal:
        levels = np.arange(n_labels, dtype=float)
        result.update({
            "ordinal_levels": list(range(n_labels)),
            "expected_level_mae": math.fsum(
                abs(float(row @ levels) - int(gold)) for row, gold in zip(rows, golds)) / n,
            "argmax_mae": float(np.abs(predictions - golds).mean()),
        })
    return result


def fit_acceptance_policy(confs, corrects, *, error_budget=0.05, min_accepted=20):
    error_budget = _unit_scalar(error_budget, "error_budget")
    min_accepted = _integer(min_accepted, "min_accepted", 1)
    confs, corrects = _confidence_correctness(confs, corrects)
    threshold, accepted, errors = _select_threshold(
        confs, corrects, error_budget, min_accepted)
    return {
        "method": _POLICY_METHOD,
        "version": _POLICY_VERSION,
        "score": _POLICY_SCORE,
        "comparison": ">=",
        "threshold": threshold,
        "dev_n": len(confs),
        "dev_accepted": accepted,
        "dev_errors": errors,
        "dev_coverage": accepted / len(confs),
        "dev_risk": errors / accepted if accepted else None,
        "error_budget": error_budget,
        "min_accepted": min_accepted,
        "guarantee": False,
        "caveat": (
            "Empirical dev risk only, not a statistical deployment risk guarantee; "
            "freeze this policy before independent test evaluation."),
    }


def _policy_threshold(policy):
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be a mapping produced by fit_acceptance_policy")
    for key, expected in (("method", _POLICY_METHOD), ("score", _POLICY_SCORE),
                          ("comparison", ">=")):
        value = policy.get(key)
        if not isinstance(value, str) or value != expected:
            raise ValueError(f"policy {key} must be {expected!r}")
    if _integer(policy.get("version"), "policy version", 1) != _POLICY_VERSION:
        raise ValueError(f"unsupported policy version; expected {_POLICY_VERSION}")
    if "threshold" not in policy:
        raise ValueError("policy must contain threshold (None means accept none)")
    threshold = policy["threshold"]
    return None if threshold is None else _unit_scalar(threshold, "policy threshold")


def _wilson_interval(errors, accepted):
    lower = upper = None
    if accepted:
        proportion = errors / accepted
        z = NormalDist().inv_cdf(0.975)
        denominator = 1 + z ** 2 / accepted
        center = (proportion + z ** 2 / (2 * accepted)) / denominator
        half_width = z * math.sqrt(
            proportion * (1 - proportion) / accepted + z ** 2 / (4 * accepted ** 2)
        ) / denominator
        lower = max(0.0, center - half_width) if errors else 0.0
        upper = min(1.0, center + half_width) if errors < accepted else 1.0
    return {
        "method": "wilson",
        "confidence_level": 0.95,
        "lower": lower,
        "upper": upper,
        "description": (
            "Descriptive binomial error-rate interval among accepted examples; "
            "assumes independent Bernoulli outcomes, not a deployment guarantee."),
    }


def evaluate_acceptance_policy(confs, corrects, policy):
    threshold = _policy_threshold(policy)
    confs, corrects = _confidence_correctness(confs, corrects, allow_empty=True)
    selected = np.zeros(len(confs), dtype=bool) if threshold is None else confs >= threshold
    accepted = int(selected.sum())
    errors = int((1 - corrects[selected]).sum())
    return {
        "n": len(confs),
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(confs) if len(confs) else None,
        "risk": errors / accepted if accepted else None,
        "risk_ci95": _wilson_interval(errors, accepted),
        "score": _POLICY_SCORE,
        "threshold": threshold,
    }
