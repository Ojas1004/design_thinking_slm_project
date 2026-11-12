"""Convert the generator's JSONL splits into Hugging Face Parquet layout.

Usage: python convert_to_hf.py <dir with energydata_sml_{train,val,test}.jsonl> <out dir>
Writes <out>/data/{train,validation,test}-00000-of-00001.parquet
"""
import sys
from pathlib import Path
from datasets import load_dataset

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "data").mkdir(parents=True, exist_ok=True)
for name, split in [("train", "train"), ("val", "validation"), ("test", "test")]:
    ds = load_dataset("json", data_files=str(src / f"energydata_sml_{name}.jsonl"), split="train")
    ds.to_parquet(str(out / "data" / f"{split}-00000-of-00001.parquet"))
    print(split, ds.num_rows)
