from pathlib import Path
import pytest
from src.data.audit import discover_root
from src.config import FOLDER_LABELS

def test_discover_root_nested(tmp_path):
 root=tmp_path/"dataset"; (root/"classification_task").mkdir(parents=True);(root/"segmentation_task").mkdir();assert discover_root(tmp_path)==root

def test_discover_root_rejects_unknown(tmp_path):
 with pytest.raises(ValueError):discover_root(tmp_path)

def test_folder_label_mapping():
 assert FOLDER_LABELS["no_tumor"]=="non_tumorous"

def test_audit_uses_histogram_and_checks_classification_integrity(tmp_path):
 from PIL import Image
 from src.data.audit import audit_dataset
 root=tmp_path/"dataset"
 for task in ("classification_task","segmentation_task"):
  for split in ("train","test"):(root/task/split).mkdir(parents=True,exist_ok=True)
 (root/"classification_task/train/glioma").mkdir()
 (root/"classification_task/train/glioma/bad.jpg").write_bytes(b"not an image")
 (root/"segmentation_task/train/images").mkdir();(root/"segmentation_task/train/masks").mkdir()
 Image.new("L",(4,4),1).save(root/"segmentation_task/train/masks/m.png")
 report=audit_dataset(root,tmp_path/"reports")
 assert len(report["classification"]["integrity_errors"])==1
 assert report["mask_unique_values"]=={"1":16}
