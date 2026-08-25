"""Convert the two CSVs to Parquet for Hugging Face upload.

CSV -> Parquet+zstd is roughly a 9x reduction (766 MB -> 88 MB) with no loss,
and loads several times faster. HF serves Parquet natively (dataset viewer,
streaming, `load_dataset`), so this is the format to publish in.

Layout matches the `configs:` block in the dataset card:

    hf_upload/
      with_ai/train-00000-of-00001.parquet
      humans_only/train-00000-of-00001.parquet
"""
import os
import pandas as pd

OUT = "hf_upload"
JOBS = [("data/ground_truth.csv", "with_ai"),
        ("data/tickets.csv", "humans_only")]

for src, cfg in JOBS:
    d = os.path.join(OUT, cfg)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "train-00000-of-00001.parquet")
    df = pd.read_csv(src, low_memory=False)
    df.to_parquet(dst, compression="zstd", index=False)
    a, b = os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6
    print(f"{src:28} {a:7.0f} MB -> {dst}  {b:6.0f} MB  ({b/a*100:.0f}%)")
    print(f"  {len(df):,} rows x {len(df.columns)} cols")

print(f"\nready to upload: {OUT}/")
