"""
Download additional PG-19 shards (shards 1, 2, 3) to scale corpus to ~5,000 books (~750M tokens).
Total download size ~900 MB, staying well under the 20 GB ceiling.
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
PG19_DIR = ROOT / "data" / "raw" / "pg19" / "data"
PG19_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "emozilla/pg19"
TARGET_FILES = [
    "data/train-00001-of-00023-2b13c68dbeba63b5.parquet",
    "data/train-00002-of-00023-8e4dd6b70d51dfcf.parquet",
    "data/train-00003-of-00023-e7f0ba0d3c2eb6c0.parquet",
]

def main():
    print(f"Downloading {len(TARGET_FILES)} additional PG-19 shards (~900 MB)...", flush=True)
    for idx, fname in enumerate(TARGET_FILES):
        out_name = Path(fname).name
        dest = PG19_DIR / out_name
        if dest.exists() and dest.stat().st_size > 100 * 1024 * 1024:
            print(f"  Shard {idx+1}/{len(TARGET_FILES)} already exists: {out_name} ({dest.stat().st_size / (1024*1024):.1f} MB)", flush=True)
            continue
        print(f"  Downloading shard {idx+1}/{len(TARGET_FILES)}: {fname}...", flush=True)
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=str(ROOT / "data" / "raw" / "pg19"),
            local_dir_use_symlinks=False
        )
        print(f"  Saved to {downloaded_path} ({Path(downloaded_path).stat().st_size / (1024*1024):.1f} MB)", flush=True)
    print("Download completed successfully!", flush=True)

if __name__ == "__main__":
    main()
