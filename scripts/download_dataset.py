import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kaggle_download import download_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--handle", default="briscdataset/brisc2025")
    p.add_argument("--output-dir", default="data/raw")
    a = p.parse_args()
    print(download_dataset(a.handle, a.output_dir))
