import argparse, json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audit import audit_dataset

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default=str(ROOT / "data/raw")); p.add_argument("--report-dir",default=str(ROOT / "reports")); a=p.parse_args()
    result=audit_dataset(a.data_root,a.report_dir); print(json.dumps({"root":result["root"],"classification":result["classification"],"segmentation":result["segmentation"],"warnings":result["warnings"]},indent=2))
