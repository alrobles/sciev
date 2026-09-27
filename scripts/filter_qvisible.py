"""Keep rows whose encoded ctx still contains the question/instruction
tail (i.e., passage truncation did not erase the typed question)."""
import json, sys, os
os.environ.setdefault("HF_HOME","/beegfs/a474r867/hf-cache")
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct",local_files_only=True)
src,dst=sys.argv[1],sys.argv[2]
rows=[json.loads(l) for l in open(src)]
kept=[r for r in rows if "Question" in tok.decode(r["ctx"][-40:])]
with open(dst,"w") as f:
    for r in kept: f.write(json.dumps(r)+"\n")
print(dst,len(rows),"->",len(kept))
