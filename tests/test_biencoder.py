import json
import sys

import pytest
import torch

from sciev import biencoder
from sciev.biencoder import BiEncoderHeads
from test_inference import TinyBackbone


def row(index, gold=0, mark=40):
    return {"qid": f"q{index}", "ctx": [index % 8 + 1, mark],
            "opts": [[index % 8 + 1, mark + 1], [index % 8 + 1, mark + 2],
                     [index % 8 + 1, mark + 3], [index % 8 + 1, mark + 4]],
            "gold": gold, "kind": "choice", "encoding": "systemone-v2",
            "schema_version": 2, "split_group": f"g{index}", "source": "arc_easy"}


def test_embeddings_are_deterministic_and_cached_by_content():
    model = TinyBackbone().eval()
    rows = [row(1), row(2)]
    cache = biencoder.embed_dataset(model, rows, "cpu", batch=2)
    again = biencoder.embed_dataset(model, rows, "cpu", batch=2, existing=dict(cache))
    assert set(cache) == set(again)
    key = biencoder.cache_key(rows[0]["ctx"])
    assert torch.equal(cache[key], again[key])
    unique = {biencoder.cache_key(seq) for seq in biencoder._row_sequences(rows)}
    assert len(cache) == len(unique)


def test_embed_resumes_from_partial_cache(tmp_path):
    rows = [row(i, mark=40) for i in range(6)]
    saved = {}

    def persist(live_cache, done, total):
        # on_batch receives embed_dataset's internal dict (a copy of
        # `existing`); snapshotting any other binding would save nothing
        if done == 2:
            saved.update({key: tensor.clone()
                          for key, tensor in live_cache.items()})

    full = biencoder.embed_dataset(TinyBackbone(), rows, "cpu", batch=2,
                                   on_batch=persist)
    assert 0 < len(saved) < len(full)
    calls = []

    class Counting(TinyBackbone):
        def hidden_layers(self, ids, layer_idx):
            calls.append(1)
            return super().hidden_layers(ids, layer_idx)

    resumed = biencoder.embed_dataset(Counting(), rows, "cpu", batch=2,
                                      existing=saved)
    assert set(resumed) == set(full)
    # cached keys carry over bit-identical; only missing keys recompute
    assert all(torch.equal(resumed[key], saved[key]) for key in saved)
    assert len(calls) == len(full) - len(saved)


def test_merge_caches_rejects_conflicting_embeddings(tmp_path):
    cache = {biencoder.cache_key([1, 2]): torch.zeros(8)}
    conflict = {biencoder.cache_key([1, 2]): torch.ones(8)}
    for index, payload in enumerate((cache, conflict)):
        biencoder.save_cache(tmp_path / f"c{index}.pt", payload, backbone="t",
                             layer=-1, source_files=[])
    with pytest.raises(ValueError, match="conflicting"):
        biencoder.merge_caches([tmp_path / "c0.pt", tmp_path / "c1.pt"])


def test_infonce_learns_to_rank_gold_above_distractors():
    torch.manual_seed(0)
    dim, k, n = 8, 4, 64
    states = torch.randn(n, dim)
    gold_actions = states + 0.05 * torch.randn(n, dim)
    distractors = torch.randn(n, k - 1, dim)
    actions = torch.cat([gold_actions.unsqueeze(1), distractors], dim=1)
    golds = torch.zeros(n, dtype=torch.long)
    heads = BiEncoderHeads(dim, proj_dim=dim, depth=1)
    opt = torch.optim.AdamW(heads.parameters(), lr=0.05)
    initial = None
    for step in range(300):
        loss = biencoder.infonce_loss(heads, states, actions, golds)
        initial = float(loss.detach()) if initial is None else initial
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < initial
    with torch.inference_mode():
        logits = torch.stack([heads.decision_logits(states[i], actions[i])
                              for i in range(n)])
    assert float((logits.argmax(1) == golds).float().mean()) > 0.9


def test_eval_vectors_protocol_schema_and_deterministic_tiebreak():
    torch.manual_seed(1)
    heads = BiEncoderHeads(8, proj_dim=8, depth=1)
    rows = [row(1), row(2)]
    vectors = [(torch.randn(8), torch.randn(4, 8), r["gold"], r) for r in rows]
    metrics = biencoder.eval_vectors(heads, vectors, return_predictions=True)
    for key in ("n", "acc", "nll", "brier", "ece"):
        assert key in metrics
    assert metrics["n"] == 2 and metrics["backbone_forwards"] == 0
    record = metrics["predictions"][0]
    for key in ("qid", "gold", "prediction", "probabilities",
                "max_probability", "correct", "split_group"):
        assert key in record
    assert abs(sum(record["probabilities"]) - 1.0) < 1e-9


def test_fit_bi_temperature_is_positive_and_finite():
    torch.manual_seed(2)
    heads = BiEncoderHeads(8, proj_dim=8, depth=1)
    vectors = [(torch.randn(8), torch.randn(4, 8), i % 4, row(i))
               for i in range(16)]
    temperature = biencoder.fit_bi_temperature(heads, vectors)
    assert 0 < temperature < 100


def test_heads_checkpoint_roundtrip(tmp_path):
    heads = BiEncoderHeads(8, proj_dim=4, depth=2)
    biencoder.save_heads(tmp_path / "decision.pt", heads, step=3,
                         training_data={"identity_hashes": []})
    loaded, checkpoint = biencoder.load_heads(tmp_path / "decision.pt")
    assert checkpoint["format"] == biencoder.CHECKPOINT_FORMAT
    state, actions = torch.randn(8), torch.randn(4, 8)
    assert torch.equal(heads.decision_logits(state, actions),
                       loaded.decision_logits(state, actions))


def test_train_cli_writes_checkpoint_and_eval_cli_reports(tmp_path, monkeypatch):
    model = TinyBackbone().eval()
    train = tmp_path / "train.jsonl"
    select = tmp_path / "select.jsonl"
    cal = tmp_path / "cal.jsonl"
    evl = tmp_path / "eval.jsonl"
    train.write_text("\n".join(json.dumps(row(i, mark=40)) for i in range(32)))
    select.write_text("\n".join(json.dumps(row(100 + i, mark=50)) for i in range(8)))
    cal.write_text("\n".join(json.dumps(row(200 + i, mark=60)) for i in range(8)))
    evl.write_text("\n".join(json.dumps(row(300 + i, mark=70)) for i in range(8)))
    cache_path = tmp_path / "embeds.pt"
    cache = {}
    for path in (train, select, cal, evl):
        from sciev.data import load_decisions_ids
        cache = biencoder.embed_dataset(model, load_decisions_ids(path), "cpu",
                                        existing=cache)
    biencoder.save_cache(cache_path, cache, backbone="tiny", layer=-1,
                         source_files=[])
    run = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "biencoder", "train", "--decisions-train", str(train),
        "--selection-dev", str(select), "--heldout", str(cal),
        "--heldout", str(evl), "--cache", str(cache_path), "--out", str(run),
        "--steps", "8", "--bs", "8", "--eval-every", "4", "--select-limit", "4",
        "--proj-dim", "8", "--device", "cpu"])
    biencoder.main()
    assert (run / "decision.pt").exists()
    progress = json.loads((run / "progress.json").read_text())
    assert progress["budget_complete"] and progress["completed_steps"] == 8
    out_dir = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", [
        "biencoder", "eval", "--heads", str(run / "decision.pt"),
        "--cache", str(cache_path), "--calibration-dev", str(cal),
        "--policy-dev", str(select), "--evaluation", f"heldout={evl}",
        "--out-dir", str(out_dir)])
    biencoder.main()
    report = json.loads((out_dir / "heldout.json").read_text())
    assert report["protocol_version"] == 2
    assert report["architecture"] == "biencoder_contrastive"
    assert report["training_overlap"]["status"] == "verified"
    assert report["decisions_eval"]["n"] == 8
    calibration = json.loads((out_dir / "calibration.json").read_text())
    assert calibration["temperature"] > 0


def test_train_allows_overlapping_heldouts_but_guards_train(tmp_path, monkeypatch):
    model = TinyBackbone().eval()
    train = tmp_path / "train.jsonl"
    select = tmp_path / "select.jsonl"
    main = tmp_path / "main.jsonl"
    diamond = tmp_path / "diamond.jsonl"
    train.write_text("\n".join(json.dumps(row(i, mark=40)) for i in range(16)))
    select.write_text("\n".join(json.dumps(row(100 + i, mark=50)) for i in range(4)))
    shared = [row(200 + i, mark=60) for i in range(4)]
    main.write_text("\n".join(json.dumps(r) for r in shared + [row(300, mark=60)]))
    diamond.write_text("\n".join(json.dumps(r) for r in shared[:2]))
    from sciev.data import load_decisions_ids
    cache = {}
    for path in (train, select, main, diamond):
        cache = biencoder.embed_dataset(model, load_decisions_ids(path), "cpu",
                                      existing=cache)
    cache_path = tmp_path / "c.pt"
    biencoder.save_cache(cache_path, cache, backbone="tiny", layer=-1,
                         source_files=[])

    def argv(*extra):
        monkeypatch.setattr(sys, "argv", ["biencoder", "train",
            "--decisions-train", str(train), "--selection-dev", str(select),
            "--cache", str(cache_path), "--steps", "2", "--device", "cpu",
            *extra])

    argv("--heldout", str(main), "--heldout", str(diamond),
         "--out", str(tmp_path / "run"))
    biencoder.main()  # diamond ⊂ main: heldout-vs-heldout overlap is fine

    poisoned = tmp_path / "poisoned.jsonl"
    poisoned.write_text(json.dumps(row(0, mark=40)))  # identical content to a train row
    argv("--heldout", str(poisoned), "--out", str(tmp_path / "run2"))
    with pytest.raises(ValueError, match="overlap"):
        biencoder.main()


def test_train_rejects_gpqa_rows(tmp_path, monkeypatch):
    model = TinyBackbone().eval()
    poisoned = tmp_path / "train.jsonl"
    select = tmp_path / "select.jsonl"
    bad = dict(row(1), source="gpqa")
    poisoned.write_text("\n".join(json.dumps(r) for r in [bad, row(2), row(3)]))
    select.write_text("\n".join(json.dumps(row(100 + i)) for i in range(4)))
    from sciev.data import load_decisions_ids
    cache = biencoder.embed_dataset(
        model, load_decisions_ids(poisoned) + load_decisions_ids(select), "cpu")
    cache_path = tmp_path / "c.pt"
    biencoder.save_cache(cache_path, cache, backbone="tiny", layer=-1,
                         source_files=[])
    monkeypatch.setattr(sys, "argv", [
        "biencoder", "train", "--decisions-train", str(poisoned),
        "--selection-dev", str(select), "--cache", str(cache_path),
        "--out", str(tmp_path / "run"), "--steps", "2", "--device", "cpu"])
    with pytest.raises(ValueError, match="GPQA"):
        biencoder.main()


def _identity_heads(dim=4):
    """Bi-encoder heads whose projections are the identity, so
    decision_logits is the raw dot product and ranks are craftable."""
    heads = BiEncoderHeads(dim, proj_dim=dim, depth=1)
    heads.state_head = torch.nn.Identity()
    heads.action_head = torch.nn.Identity()
    return heads


def test_recall_row_mode_known_ranks():
    heads = _identity_heads()
    e = torch.eye(4)
    vectors = [
        (e[0], torch.stack([e[0], e[1], e[2]]), 0, row(1)),     # rank 1
        (e[0], torch.stack([2 * e[0], e[0], e[1]]), 1, row(2)),  # rank 2
        (e[0], torch.stack([2 * e[0], 3 * e[0], e[0]]), 2, row(3))]  # rank 3
    metrics = biencoder.recall_at_k(heads, vectors, ks=[1, 2, 4], pool="row")
    assert metrics["n"] == 3 and metrics["pool"] == "row"
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@2"] == pytest.approx(2 / 3)
    assert metrics["recall@4"] == 1.0
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 1 / 3) / 3)


def test_recall_suite_mode_dedupes_shared_actions():
    heads = _identity_heads()
    e = torch.eye(4)
    shared = torch.stack([e[1], e[2]])  # same label space across rows
    vectors = [(e[1], shared, 0, row(1)), (e[2], shared, 1, row(2))]
    metrics = biencoder.recall_at_k(heads, vectors, ks=[1, 2], pool="suite")
    assert metrics["pool_size"] == 2
    assert metrics["recall@1"] == 1.0 and metrics["mrr"] == 1.0


def test_recall_rejects_empty_and_bad_pool():
    heads = _identity_heads()
    with pytest.raises(ValueError, match="at least one row"):
        biencoder.recall_at_k(heads, [], ks=[1])
    e = torch.eye(4)
    vectors = [(e[0], torch.stack([e[0], e[1]]), 0, row(1))]
    with pytest.raises(ValueError, match="unknown pool"):
        biencoder.recall_at_k(heads, vectors, ks=[1], pool="bogus")


def test_recall_cli_writes_report(tmp_path, monkeypatch):
    model = TinyBackbone().eval()
    rows = [row(i, mark=50) for i in range(4)]
    cache = biencoder.embed_dataset(model, rows, "cpu")
    cache_path = tmp_path / "embeds.pt"
    biencoder.save_cache(cache_path, cache, backbone="tiny", layer=-1,
                         source_files=[])
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("".join(json.dumps(r) + "\n" for r in rows))
    heads_path = tmp_path / "decision.pt"
    biencoder.save_heads(heads_path, BiEncoderHeads(cache[next(iter(cache))].numel(),
                                                   proj_dim=8, depth=1))
    out = tmp_path / "recall.json"
    monkeypatch.setattr(sys, "argv", [
        "biencoder", "recall", "--heads", str(heads_path),
        "--cache", str(cache_path), "--decisions", str(decisions),
        "--pool", "row", "--out", str(out)])
    biencoder.main()
    report = json.loads(out.read_text())
    result = report["results"][decisions.name]
    assert result["n"] == 4 and result["recall@4"] == 1.0
    assert report["protocol_version"] == 2 and report["role"] == "diagnostic"
    with pytest.raises(FileExistsError):
        biencoder.main()  # anti-overwrite, consistent with eval
