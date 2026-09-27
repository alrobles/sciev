"""Drop rows sharing an identical effective input (ctx+opts) with
conflicting gold indices; report counts and affected qids."""
import json, sys, collections
src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)]
groups = collections.defaultdict(list)
for i, r in enumerate(rows):
    sig = (tuple(r["ctx"]), tuple(tuple(o) for o in r["opts"]))
    groups[sig].append(i)
drop, conflicts = set(), []
for sig, idxs in groups.items():
    golds = {rows[i]["gold"] for i in idxs}
    if len(idxs) > 1:
        conflicts.append({"idxs": idxs, "golds": sorted(golds),
                          "qids": [rows[i].get("qid") for i in idxs]})
        if len(golds) > 1:
            drop.update(idxs[1:])   # keep first, drop later conflicting rows
        else:
            drop.update(idxs[1:])   # exact duplicate record
kept = [r for i, r in enumerate(rows) if i not in drop]
with open(dst, "w") as f:
    for r in kept: f.write(json.dumps(r) + "\n")
rep = dst.replace(".jsonl", ".conflicts.json")
json.dump({"src": src, "n_in": len(rows), "n_out": len(kept),
           "n_dropped": len(drop), "conflicts": conflicts}, open(rep, "w"), indent=1)
print(dst, len(rows), "->", len(kept), "dropped", len(drop))
