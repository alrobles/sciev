"""Data loaders.

Two input shapes:
1. ecoreasoner pairs — pairs_L*.jsonl, one {"ctx","ok","bad"} of token ids per
   line (the L0-L3 discrimination battery format).
2. System-One JSONL — one request per line, API-shaped plus a "label" per
   question (same format kev trains on):

   {"state": "...", "questions": {"team": {"type": "choice",
        "instructions": "...", "criteria": {"a": "...", "b": "..."},
        "label": "a"}}}
"""
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path


def load_pairs(path, max_ctx=None, max_cand=None):
    """pairs jsonl -> [(ctx_ids, ok_ids, bad_ids)]."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ctx = rec["ctx"][:max_ctx] if max_ctx else rec["ctx"]
        ok = rec["ok"][:max_cand] if max_cand else rec["ok"]
        bad = rec["bad"][:max_cand] if max_cand else rec["bad"]
        if len(ctx) >= 2 and ok and bad:
            pairs.append((ctx, ok, bad))
    return pairs


def load_pairs_dir(pairs_dir):
    """Directory of pairs_L{0..3}.jsonl -> {level: pairs}."""
    out = {}
    for fp in sorted(Path(pairs_dir).glob("pairs_L*.jsonl")):
        lvl = fp.stem.split("_L")[-1]
        if lvl in ("0", "1", "2", "3"):
            out[f"L{lvl}"] = load_pairs(fp)
    return out


def load_decisions_ids(path):
    """K-way labeled decisions -> list of row dicts.

    Format: {"ctx":[int...], "opts":[[int...]...], "gold":int,
             "qid"?:str, "soft"?:[float...]}
    Emitted by data/build_toolcall_decisions.py; `soft` added by
    data/teacher_label.py joins.
    """
    from .decisions import validate_decision_row

    rows = []
    for line_number, rec in read_jsonl(path):
        try:
            row = validate_decision_row(rec)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        row.setdefault("qid", None)
        row.setdefault("soft", None)
        rows.append(row)
    return rows


def load_soft_labels(path):
    """Teacher distributions -> {qid: [float...]}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("qid") and rec.get("soft"):
            out[rec["qid"]] = rec["soft"]
    return out


def iter_decisions(path):
    """System-One JSONL -> (state, qid, question_dict, label)."""
    for line_number, rec in read_jsonl(path):
        yield from _request_decisions(rec, f"{path}:{line_number}")


def split_dev_test(path, dev_frac=0.5, seed=7331):
    """Deterministic dev/test split over decisions (dev is for temperature
    fitting; test is read once for the reported number)."""
    import random
    validate_fractions(dev_frac, 0.0)
    records = []
    for line_number, rec in read_jsonl(path):
        list(_request_decisions(rec, f"{path}:{line_number}"))
        records.append(rec)
    grouped = group_records(records, "state", scope="state")
    splits = partition_groups(grouped, random.Random(seed), dev_frac, 0.0)
    return tuple([row for rec in splits[tag] for row in _request_decisions(rec, str(path))]
                 for tag in ("dev", "train"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield line_number, rec


def _request_decisions(rec, location):
    if "state" not in rec or not isinstance(rec.get("questions"), dict):
        raise ValueError(f"{location}: expected state and a questions object")
    from .decisions import render_value
    try:
        state = render_value(rec["state"], "state", allow_empty=True)
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc
    for qid, question in rec["questions"].items():
        if not isinstance(question, dict):
            raise ValueError(f"{location}: question {qid!r} must be an object")
        if "label" in question:
            yield state, qid, question, question["label"]


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def normalized_content(value):
    text = value if isinstance(value, str) else stable_json(value)
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_fingerprint(value):
    return hashlib.sha256(normalized_content(value).encode("utf-8")).hexdigest()


DOCUMENT_KEYS = ("document_id", "doc_id", "paper_id", "corpus_id", "doi", "pmid", "pmcid")
GROUP_KEYS = ("split_group", "source_group", "group_id", "group", "pid")


def group_records(records, text_key, scope="passage", source="unknown"):
    rows = [dict(row) for row in records]
    parents = list(range(len(rows)))
    seen, tokens_by_row, document_rows = {}, [], set()

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(rows):
        row.setdefault("source", source)
        row["content_hash"] = content_fingerprint(row[text_key])
        tokens = {("content", row["content_hash"])}
        containers = [row, *row.get("questions", {}).values()]
        containers += [c["meta"] for c in containers
                       if isinstance(c, dict) and isinstance(c.get("meta"), dict)]
        for container in containers:
            for key in (*DOCUMENT_KEYS, *GROUP_KEYS):
                value = container.get(key)
                if value is None or value == "":
                    continue
                if key in DOCUMENT_KEYS:
                    document_rows.add(index)
                    namespace = "document"
                elif key == "pid":
                    namespace = "pid:" + stable_json(container.get("source", row["source"]))
                else:
                    namespace = "group"
                tokens.add((namespace, stable_json(value)))
        tokens_by_row.append(tokens)
        for token in tokens:
            if token in seen:
                parents[root(index)] = root(seen[token])
            else:
                seen[token] = index
    components = {}
    for index, tokens in enumerate(tokens_by_row):
        component = components.setdefault(root(index), {"tokens": set(), "document": False})
        component["tokens"].update(tokens)
        component["document"] |= index in document_rows
    for index, row in enumerate(rows):
        component = components[root(index)]
        group_scope = "document" if component["document"] else scope
        identity = hashlib.sha256(stable_json(sorted(component["tokens"])).encode("utf-8")).hexdigest()
        row["split_group"] = f"{group_scope}:{identity}"
        row.setdefault("group_id", row["split_group"])
        row["group_scope"] = group_scope
        row.setdefault("pid", f"{scope}:{row['content_hash']}")
    return rows


def validate_fractions(dev_frac, eval_frac):
    for name, value in (("dev_frac", dev_frac), ("eval_frac", eval_frac)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if dev_frac + eval_frac > 1:
        raise ValueError("dev_frac + eval_frac must not exceed 1")


def partition_groups(records, rng, dev_frac=0.15, eval_frac=0.20, counts=None, exclusions=None):
    validate_fractions(dev_frac, eval_frac)
    groups = sorted({row["split_group"] for row in records})
    rng.shuffle(groups)
    if counts is None:
        sizes = {"dev": int(len(groups) * dev_frac), "eval": int(len(groups) * eval_frac)}
        sizes["train"] = len(groups) - sum(sizes.values())
    else:
        if set(counts) != {"train", "dev", "eval"} or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError("split counts must specify nonnegative integer train/dev/eval group counts")
        if sum(counts.values()) > len(groups):
            raise ValueError(f"split counts exceed the {len(groups)} available source/content groups")
        sizes = counts
    owners, offset = {}, 0
    for tag in ("eval", "dev", "train"):
        owners.update({group: tag for group in groups[offset:offset + sizes[tag]]})
        offset += sizes[tag]
    splits = {tag: [] for tag in ("train", "dev", "eval")}
    for row in records:
        tag = owners.get(row["split_group"])
        if tag is None:
            exclude_record(exclusions, row, "unassigned_group")
        else:
            splits[tag].append(dict(row, split=tag))
    return splits


def record_metadata(record, source, reasoning_type):
    keys = (*DOCUMENT_KEYS, *GROUP_KEYS, "source", "source_file", "source_line", "source_row",
            "source_id", "claim_id", "group_scope", "content_hash", "sample_id", "filter_policy",
            "duplicate_sources", "meta", "split")
    metadata = {key: record[key] for key in keys if key in record}
    metadata.setdefault("source", source)
    metadata["reasoning_type"] = record.get("reasoning_type", record.get("type", reasoning_type))
    return metadata


def exclude_record(exclusions, record, reason, **details):
    if exclusions is not None:
        keys = ("pid", "source", "source_file", "source_line", "source_row", "sample_id", "split_group")
        exclusions.append({**{key: record[key] for key in keys if key in record},
                           "reason": reason, **details})


def add_decision(id_rows, text_by_state, tok, state, qid, question, gold, metadata,
                 max_ctx=640, max_opt=120, overflow="error", exclusions=None):
    from .decisions import InputOverflow, encode_question, validate_decision_row
    if overflow not in ("error", "exclude", "truncate"):
        raise ValueError("overflow must be error, exclude, or truncate")
    try:
        encoded = encode_question(tok, state, question, max_ctx=max_ctx, max_opt=max_opt,
                                  overflow="error" if overflow == "exclude" else overflow)
        decision_id = hashlib.sha256(stable_json(encoded).encode("utf-8")).hexdigest()
        qid = f"{qid}_{decision_id[:16]}"
        row = validate_decision_row({**metadata, **encoded, "gold": gold, "qid": qid, "decision_id": decision_id})
    except InputOverflow:
        if overflow != "exclude":
            raise
        exclude_record(exclusions, metadata, "input_overflow",
                       qid=qid, kind=question.get("type"))
        return None
    except ValueError as exc:
        raise ValueError(f"{qid}: {exc}") from exc
    text_meta = {**metadata, "encoding": row["encoding"], "schema_version": row["schema_version"]}
    key = (stable_json(state), metadata.get("sample_id"), metadata.get("split_group"))
    request = text_by_state.setdefault(key, {**text_meta, "state": state, "questions": {}})
    if qid in request["questions"]:
        raise ValueError(f"duplicate decision {qid!r}; deduplicate source records before encoding")
    id_rows.append(row)
    request["questions"][qid] = {**text_meta, **question, "kind": row["kind"],
                                 "option_keys": row["option_keys"], "decision_id": decision_id}


def _evidence_span(span, state, location):
    if (not isinstance(span, (list, tuple)) or len(span) != 2
            or any(type(offset) is not int for offset in span)
            or not 0 <= span[0] < span[1] <= len(state)
            or not state[span[0]:span[1]].strip()):
        raise ValueError(f"{location}: evidence_span must select nonempty evidence with integer character offsets")
    return tuple(span)


def _evidence_source(record, index):
    location = f"record {index}"
    if not isinstance(record, dict) or not isinstance(record.get("state"), str):
        raise ValueError(f"{location}: text request state must be a string")
    span = _evidence_span(record.get("evidence_span"), record["state"], location)
    group = record.get("split_group", record.get("group_id"))
    if not (isinstance(group, str) and group.strip() or type(group) is int):
        raise ValueError(f"{location}: source split_group or group_id must be a nonempty string or integer")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{location}: questions must be a nonempty object")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid.strip() or not isinstance(question, dict):
            raise ValueError(f"{location}: questions require nonempty string qids and object values")
        if "evidence_span" in question and _evidence_span(question["evidence_span"], record["state"], location) != span:
            raise ValueError(f"{location}: question {qid!r} evidence_span differs from the request")
        for key in ("split_group", "group_id", "split"):
            if key in question and key in record and question[key] != record[key]:
                raise ValueError(f"{location}: question {qid!r} {key} differs from the request")
    for container in (record, *questions.values()):
        if "evidence_control" in container:
            raise ValueError(f"{location}: evidence controls require original, not already controlled, records")
        if "ctx" in container or "opts" in container:
            raise ValueError(f"{location}: evidence controls accept text requests, not cached token encodings")
    return group, span, record["state"][span[0]:span[1]]


def _mark_evidence_control(container, marker, span):
    for field in ("label", "gold", "soft", "label_status", "decision_id", "content_hash", "truncation", "evidence_regime"):
        if field in container:
            container[f"reference_{field}"] = container.pop(field)
    container["evidence_span"] = list(span)
    container["evidence_control"] = dict(marker)
    container["label_status"] = "evidence_control"


def make_evidence_controls(records, mode="empty", seed=7331):
    from copy import deepcopy
    import random

    if mode not in ("empty", "shuffle"):
        raise ValueError("mode must be empty or shuffle")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    try:
        records = list(records)
    except TypeError as exc:
        raise ValueError("records must be an iterable of text requests") from exc
    if not records:
        raise ValueError("records must not be empty")
    sources = [_evidence_source(record, index) for index, record in enumerate(records)]
    pool, keys = {}, []
    for index, (record, (group, span, evidence)) in enumerate(zip(records, sources)):
        key = (stable_json(record.get("split")), stable_json(group), evidence)
        pool.setdefault(key, index)
        keys.append(key)
    rng, donors, controls = random.Random(seed), {}, []
    for index, (record, (group, span, evidence)) in enumerate(zip(records, sources)):
        donor_index, donor_group, replacement = None, None, ""
        if mode == "shuffle":
            key = keys[index]
            if key not in donors:
                candidates = [i for (split, donor, text), i in pool.items()
                              if split == key[0] and donor != key[1] and text != evidence]
                if not candidates:
                    raise ValueError(f"record {index}: no different-evidence donor from a different source group in the same split")
                donors[key] = rng.choice(candidates)
            donor_index = donors[key]
            donor_group, _, replacement = sources[donor_index]
        start, end = span
        controlled = deepcopy(record)
        controlled["state"] = record["state"][:start] + replacement + record["state"][end:]
        new_span = (start, start + len(replacement))
        marker = {"mode": mode, "seed": seed, "source_group": group, "donor_group": donor_group,
                  "source_record_index": index, "donor_record_index": donor_index,
                  "label_semantics": "original_reference_not_relabelled"}
        _mark_evidence_control(controlled, marker, new_span)
        if "qid" in controlled:
            controlled["reference_qid"] = controlled.pop("qid")
        questions = controlled.pop("questions")
        controlled["questions"] = {}
        for reference_qid, question in questions.items():
            identity = {"state": controlled["state"], "mode": mode, "source_group": group,
                        "source_record_index": index, "reference_qid": reference_qid,
                        "question": {key: question[key] for key in ("type", "instructions", "criteria") if key in question}}
            digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
            qid = f"{reference_qid}__evidence_{mode}_{digest[:16]}"
            _mark_evidence_control(question, marker, new_span)
            question.update(qid=qid, reference_qid=reference_qid)
            controlled["questions"][qid] = question
        controls.append(controlled)
    return controls


def input_fingerprints(paths):
    inputs = []
    for path in sorted({Path(path) for path in paths}, key=str):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        inputs.append({"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
    return inputs


def versioned_path(path):
    from .decisions import ENCODING_VERSION
    path = Path(path)
    return path if path.parent.name == ENCODING_VERSION else path.parent / ENCODING_VERSION / path.name


def write_dataset(files, manifest_path, manifest):
    from .decisions import ENCODING_VERSION
    paths = {versioned_path(path): rows for path, rows in files.items()}
    manifest_path = versioned_path(manifest_path)
    for path in (*paths, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new output location")
    payloads = {path: "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
                for path, rows in paths.items()}
    report = {**manifest, "encoding": ENCODING_VERSION, "schema_version": 2,
              "overflow": "error", "exclusions": manifest.get("exclusions", [])}
    report["counts"] = {**manifest.get("counts", {}),
                        "output_rows": {path.name: len(rows) for path, rows in paths.items()},
                        "exclusions_by_reason": dict(Counter(e["reason"] for e in report["exclusions"]))}
    report["outputs"] = [{"path": str(path), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
                         for path, payload in payloads.items()]
    payloads[manifest_path] = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    for path, payload in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    return report
