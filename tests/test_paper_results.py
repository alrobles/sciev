import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def table_before_label(label):
    text = (ROOT / "paper/main.tex").read_text()
    prefix = text[:text.index(r"\label{" + label + "}")]
    return prefix[prefix.rindex(r"\begin{tabular}"):]


def test_public_table_matches_archived_results():
    reports = json.loads((ROOT / "paper/results.json").read_text())["reports"]
    table = table_before_label("tab:public")
    rows = [line for line in table.splitlines()
            if " & " in line and not line.startswith("benchmark")]
    keys = ["c_choice_gpqa_main", "c_choice_gpqa_diamond", "c_noul_scifact",
            "c_score_scifact", "c_choice_enron", "c_choice_sst2",
            "c_choice_agnews", "c_choice_banking77"]
    assert len(rows) == len(keys)
    for row, key in zip(rows, keys):
        metrics = reports[key]["metrics"]
        cells = row.split("&")
        assert int(cells[2].strip()) == metrics["n"]
        values = [float(value) for value in re.findall(r"\b\d+\.\d{4}\b", row)]
        assert values == [metrics[name] for name in ("acc", "ece", "automation_5pct")]


def test_candidate_table_matches_archived_results():
    reports = json.loads((ROOT / "paper/results.json").read_text())["reports"]
    table = table_before_label("tab:candidates")
    labels = {"eb": r"bw1\_sr", "c": "LLaDA-8B frozen",
              "dag": "DAPT g2000", "da": "DAPT g5000"}
    for family, label in labels.items():
        rows = [line for line in table.splitlines() if line.startswith(label)]
        assert len(rows) == 1
        elite = "" if family == "c" else "_elite"
        suffixes = [f"choice{elite}", f"noul{elite}", f"score{elite}",
                    "choice_gpqa_main", "choice_gpqa_diamond",
                    "noul_scifact", "score_scifact"]
        expected = [reports[f"{family}_{suffix}"]["metrics"]["acc"]
                    for suffix in suffixes]
        values = [float(value) for value in re.findall(r"\b\d+\.\d{4}\b", rows[0])]
        assert values == expected
