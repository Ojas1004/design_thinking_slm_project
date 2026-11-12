"""Convert the generator's JSONL splits into Hugging Face Parquet layout.

Usage: python convert_to_hf.py <dir with energydata_sml_{train,val,test}.jsonl> <out dir>
Writes <out>/data/{train,validation,test}-00000-of-00001.parquet with typed
features: `task` and `label` become ClassLabel columns (a missing label becomes "none").
"""
import json
import sys
from pathlib import Path
from datasets import ClassLabel, Dataset, Features, Value

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "data").mkdir(parents=True, exist_ok=True)
splits = {"train": "train", "validation": "val", "test": "test"}

rows = {}
for split, name in splits.items():
    with open(src / f"energydata_sml_{name}.jsonl", encoding="utf-8") as fh:
        rows[split] = [json.loads(line) for line in fh]
    for r in rows[split]:
        r["label"] = r["label"] or "none"

tasks = sorted({r["task"] for rs in rows.values() for r in rs})
labels = sorted({r["label"] for rs in rows.values() for r in rs})
features = Features({
    "prompt": Value("string"),
    "completion": Value("string"),
    "task": ClassLabel(names=tasks),
    "answerable": Value("bool"),
    "label": ClassLabel(names=labels),
})

for split, rs in rows.items():
    data = {k: [r[k] for r in rs] for k in features}
    ds = Dataset.from_dict(data, features=features)
    ds.to_parquet(str(out / "data" / f"{split}-00000-of-00001.parquet"))
    print(split, ds.num_rows)
print("tasks:", tasks)
print("labels:", labels)
