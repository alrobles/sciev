import argparse
import json
from pathlib import Path

from sciev.data import load_decisions_ids
from sciev.protocol import assert_disjoint_splits, file_fingerprint


def compact(report):
    return {key: value for key, value in report.items() if key != "splits"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--gpqa-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    qa, gpqa = Path(args.qa_dir), Path(args.gpqa_dir)
    paths = {name: qa / f"qa_choice_{name}.jsonl" for name in ("train", "select", "calibration", "policy", "eval")}
    rows = {name: load_decisions_ids(path) for name, path in paths.items()}
    checks = compact(assert_disjoint_splits(rows))
    for name, entries in rows.items():
        if any(len(row["opts"]) != 4 or row["truncation"]["context_tokens"] or
               any(row["truncation"]["option_tokens"]) for row in entries):
            raise ValueError(f"unexpected option count or truncation in {name}")
        manifest = json.loads(paths[name].with_suffix(".manifest.json").read_text())
        if manifest.get("builder_version") != 3:
            raise ValueError("training requires the layout-preserving QA builder v3")
        declared = {entry["path"]: entry["sha256"] for entry in manifest["outputs"]}
        if declared.get(str(paths[name])) != file_fingerprint(paths[name])["sha256"]:
            raise ValueError(f"output digest disagrees with manifest for {name}")
    report = {"qa_disjointness": checks, "counts": {name: len(entries) for name, entries in rows.items()},
              "inputs": {name: file_fingerprint(path) for name, path in paths.items()}, "gpqa_checks": {}}
    for name, expected in (("main", 440), ("diamond", 193)):
        path = gpqa / f"gpqa_{name}_choice_eval_clean.jsonl"
        external = load_decisions_ids(path)
        if len(external) != expected:
            raise ValueError(f"unexpected GPQA {name} cohort size")
        report["gpqa_checks"][name] = compact(assert_disjoint_splits({**rows, "gpqa": external}))
        report["inputs"][f"gpqa_{name}"] = file_fingerprint(path)
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
