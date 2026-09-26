import hashlib
import json
import re
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

from . import decisions


_IDENTIFIER_NAMESPACES = {
    "qid": "qid", "reference_qid": "qid",
    "sample_id": "sample", "source_sample_id": "sample", "reference_sample_id": "sample",
    "decision_id": "decision", "source_decision_id": "decision", "reference_decision_id": "decision",
    "split_group": "group", "source_group": "group", "group_id": "group", "group": "group",
    "donor_group": "group", "negative_source_group": "group",
    "content_hash": "content", "source_content_hash": "content", "reference_content_hash": "content",
    "donor_content_hash": "content", "content_id": "content_id", "source_content_id": "content_id",
    "document_id": "document", "doc_id": "document", "paper_id": "document", "corpus_id": "document",
    "source_document_id": "document", "source_doc_id": "document",
    "doi": "doi", "source_doi": "doi", "pmid": "pmid", "source_pmid": "pmid",
    "pmcid": "pmcid", "source_pmcid": "pmcid",
    "pid": "pid", "source_pid": "pid", "donor_pid": "pid",
    "source_id": "source_id", "claim_id": "claim_id", "source_claim_id": "claim_id",
}
_IDENTIFIER_LISTS = {
    "negative_source_groups": "group", "source_groups": "group",
    "document_ids": "document", "doc_ids": "document", "paper_ids": "document", "corpus_ids": "document",
}
_METADATA_CONTAINERS = (
    "meta", "metadata", "provenance", "source_provenance", "negative_provenance",
    "negative_sources", "duplicate_sources", "evidence_control",
)
_DOCUMENT_NAMESPACES = {"document", "doi", "pmid", "pmcid"}
_LOCAL_NAMESPACES = {"pid", "source_id", "claim_id"}


def scientific_recipe(decision_type):
    if not isinstance(decision_type, str) or decision_type not in ("choice", "noul", "score"):
        raise ValueError("decision_type must be choice, noul, or score")
    return {"freeze": True, "r2_mode": "spanpool", "canonical_order": True,
            "orders": 1, "head_kind": "attnpool" if decision_type == "choice" else "mlp",
            "steps": 2000 if decision_type == "choice" else 3000, "head_lr": 3e-4,
            "warmup": 200, "accum": 1, "ordinal": 1.0 if decision_type == "score" else 0.0}


def file_fingerprint(path):
    path = Path(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def _digest(namespace, value):
    payload = json.dumps(["reverse_jev.protocol", 1, namespace, value], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scope():
    return {"exact_inputs": True, "declared_identifiers": True,
            "document_disjointness": "not_proven", "paraphrase_leakage": "not_checked"}


def _coverage(count, total):
    return "available" if count == total else "partial" if count else "not_available"


def _identifier(value, field):
    if value is None or isinstance(value, str) and value == "":
        return None
    if isinstance(value, str) and value.strip():
        return value
    if not isinstance(value, bool) and isinstance(value, Integral):
        return int(value)
    raise ValueError(f"{field} must be a nonempty string or integer identifier")


def _metadata_identities(row):
    identities, namespaces, document_ids, source_ids, negative_ids, active = (set() for _ in range(6))

    def add(namespace, value, source, negative):
        if value is None:
            return
        payload = [source, value] if namespace in _LOCAL_NAMESPACES else value
        identity = _digest(namespace, payload)
        identities.add(identity)
        namespaces.add(namespace)
        if namespace not in {"qid", "decision"}:
            source_ids.add(identity)
        if namespace in _DOCUMENT_NAMESPACES and not negative:
            document_ids.add(identity)
        if negative:
            negative_ids.add(identity)

    def visit(container, inherited_source=None, negative=False):
        if id(container) in active:
            raise ValueError("provenance metadata must not contain cycles")
        active.add(id(container))
        source = _identifier(container.get("source", inherited_source), "source")
        for field, namespace in _IDENTIFIER_NAMESPACES.items():
            value = _identifier(container.get(field), field)
            add(namespace, value, source, negative or field.startswith(("negative_", "donor_")))
        for field, namespace in _IDENTIFIER_LISTS.items():
            values = container.get(field)
            if values is None:
                continue
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{field} must be a list of identifiers")
            for value in values:
                add(namespace, _identifier(value, field), source, negative or field.startswith("negative_"))
        source_file = _identifier(container.get("source_file"), "source_file")
        if source_file is not None:
            for field in ("source_line", "source_row"):
                coordinate = _identifier(container.get(field), field)
                if coordinate is not None:
                    add("source_location", [source_file, field, coordinate], source, negative)
        for field in _METADATA_CONTAINERS:
            child = container.get(field)
            is_negative = negative or field.startswith("negative_")
            if isinstance(child, Mapping):
                visit(child, source, is_negative)
            elif isinstance(child, (list, tuple)):
                for entry in child:
                    if isinstance(entry, Mapping):
                        visit(entry, source, is_negative)
                    elif field == "negative_sources":
                        add("group", _identifier(entry, field), source, True)
                    else:
                        raise ValueError(f"{field} must contain provenance objects")
            elif child is not None and field not in ("meta", "metadata"):
                raise ValueError(f"{field} must contain provenance objects")
        active.remove(id(container))

    visit(row)
    return identities, namespaces, bool(document_ids), bool(source_ids), bool(negative_ids)


def _semantic_options(row):
    kind, keys, labels, opts = row.get("kind"), row.get("option_keys"), row.get("label_space"), row["opts"]
    if labels is not None:
        if (not isinstance(labels, (list, tuple)) or len(labels) != len(opts)
                or any(not isinstance(label, str) or not label.strip() for label in labels)
                or len(set(labels)) != len(labels)
                or keys is not None and set(keys) != set(labels)):
            raise ValueError("label_space must name each fixed semantic option exactly once")
        labels = list(labels)
    if kind not in ("noul", "score") and labels is None and row.get("fixed_labels") is not True:
        return None
    if kind == "noul" and len(opts) != 2:
        raise ValueError("noul requires exactly two options")
    if labels is None:
        if keys is None:
            labels = list(range(len(opts)))
        elif kind == "score" and set(keys) == {str(index) for index in range(len(opts))}:
            labels = [str(index) for index in range(len(opts))]
        elif kind == "noul" and set(keys) == {"true", "false"}:
            labels = ["true", "false"]
        else:
            labels = list(keys)
    ordered = [opts[keys.index(label)] for label in labels] if keys is not None else opts
    return {"kind": kind, "labels": labels, "opts": ordered}


def dataset_contract(rows):
    if isinstance(rows, (Mapping, str, bytes)):
        raise ValueError("rows must be a nonempty iterable of decision objects")
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise ValueError("rows must be a nonempty iterable of decision objects") from exc
    encodings, kinds, counts, identities, inputs, effective_inputs, namespaces = (set() for _ in range(7))
    gold_by_input = {}
    n_rows = identified_rows = source_rows = document_rows = negative_rows = 0
    for index, original in enumerate(iterator, 1):
        try:
            qid = original.get("qid") if isinstance(original, Mapping) else None
            if isinstance(qid, Integral) and not isinstance(qid, bool):
                row = decisions.validate_decision_row(dict(original, qid=str(int(qid))))
                row["qid"] = int(qid)
            else:
                row = decisions.validate_decision_row(original)
            encoding = row.get("encoding", "legacy_ids")
            if not isinstance(encoding, str) or not encoding.strip():
                raise ValueError("encoding must be a nonempty declared string")
            encodings.add(encoding)
            if len(encodings) > 1:
                raise ValueError("dataset cannot mix encoding profiles")
            input_identity = _digest("input", [row["ctx"], sorted(row["opts"])])
            effective_identity = _digest("effective_input", [input_identity, _semantic_options(row)])
            gold_identity = _digest("gold_option", row["opts"][row["gold"]])
            if effective_identity in gold_by_input and gold_by_input[effective_identity] != gold_identity:
                raise ValueError("identical effective input has conflicting gold option identity")
            gold_by_input[effective_identity] = gold_identity
            row_ids, row_namespaces, has_document, has_source, has_negative = _metadata_identities(row)
        except ValueError as exc:
            raise ValueError(f"row {index}: {exc}") from exc
        n_rows += 1
        identified_rows += bool(row_ids)
        source_rows += has_source
        document_rows += has_document
        negative_rows += has_negative
        if row.get("kind") is not None:
            kinds.add(row["kind"])
        counts.add(len(row["opts"]))
        inputs.add(input_identity)
        effective_inputs.add(effective_identity)
        identities.update((input_identity, effective_identity))
        identities.update(row_ids)
        namespaces.update(row_namespaces)
    if not n_rows:
        raise ValueError("dataset requires at least one decision; rows are empty")
    return {"contract_version": 1, "encoding": next(iter(encodings)), "kinds": sorted(kinds),
            "n_rows": n_rows, "option_counts": sorted(counts), "identity_hashes": sorted(identities),
            "input_hashes": sorted(inputs), "effective_input_hashes": sorted(effective_inputs),
            "provenance_status": _coverage(source_rows, n_rows),
            "provenance": {"rows_with_identifiers": identified_rows,
                           "rows_with_source_provenance": source_rows,
                           "rows_with_document_ids": document_rows,
                           "rows_with_negative_provenance": negative_rows,
                           "identifier_namespaces": sorted(namespaces),
                           "document_status": _coverage(document_rows, n_rows)},
            "scope": _scope()}


def assert_disjoint_splits(named_rows):
    if not isinstance(named_rows, Mapping) or not named_rows:
        raise ValueError("named_rows must be a nonempty mapping of split roles to decisions")
    if any(not isinstance(name, str) or not name.strip() for name in named_rows):
        raise ValueError("split roles must be nonempty strings")
    contracts = {name: dataset_contract(rows) for name, rows in named_rows.items()}
    previous = {}
    pairs_checked = 0
    for name, contract in contracts.items():
        identities = set(contract["identity_hashes"])
        for other, other_identities in previous.items():
            overlap = identities & other_identities
            if overlap:
                raise ValueError(f"identity overlap between {other!r} and {name!r}: {len(overlap)} known identities")
            pairs_checked += 1
        previous[name] = identities
    return {"status": "verified", "reason": "no_known_identity_overlap", "overlap_count": 0,
            "pairs_checked": pairs_checked, "splits": contracts, "scope": _scope()}


def assert_checkpoint_disjoint(checkpoint, rows, role="evaluation"):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be an object")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a nonempty string")
    contract = dataset_contract(rows)
    training = checkpoint.get("training_data")
    if training is not None and not isinstance(training, Mapping):
        raise ValueError("checkpoint training_data must be an object")
    report = {"role": role, "data": contract, "data_identity_count": len(contract["identity_hashes"]),
              "scope": _scope()}
    if training is None or "identity_hashes" not in training:
        return {**report, "status": "unverified", "reason": "training_identities_not_available",
                "training_identity_status": "not_available", "training_identity_count": None,
                "overlap_count": None}
    hashes = training["identity_hashes"]
    if (not isinstance(hashes, (list, tuple)) or not hashes
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
                   for value in hashes)):
        raise ValueError("checkpoint training_data.identity_hashes must be a nonempty list of SHA256 strings")
    if training.get("contract_version", 1) != 1:
        raise ValueError("checkpoint training_data contract_version is unsupported")
    training_identities = {value.lower() for value in hashes}
    overlap = training_identities.intersection(contract["identity_hashes"])
    if overlap:
        raise ValueError(f"training/{role} identity overlap: {len(overlap)} known identities")
    return {**report, "status": "verified", "reason": "no_known_identity_overlap",
            "training_identity_status": "available", "training_identity_count": len(training_identities),
            "overlap_count": 0}


def ensure_fresh_output(path):
    path = Path(path)
    if (path.exists() or path.is_symlink()) and not path.is_dir():
        raise FileExistsError(f"output path already exists and is not a directory: {path}")
    for name in ("decision.pt", "model.pt", "training_config.json", "temperature.json"):
        artifact = path / name
        if artifact.exists() or artifact.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing training artifact: {artifact}")
