import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.kaggle_download import download_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--handle", default="briscdataset/brisc2025")
    p.add_argument("--output-dir", default=str(ROOT / "data/raw"))
    a = p.parse_args()
    print(download_dataset(a.handle, a.output_dir))
