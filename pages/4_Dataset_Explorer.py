import json
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="Dataset audit | NeuroScan",layout="wide");st.title("Dataset audit")
p=Path("reports/dataset_audit.json")
if p.exists():
 d=json.loads(p.read_text());st.write(f"Audited root: `{d['root']}`");st.subheader("Classification counts");st.json(d.get("classification",{}));st.subheader("Segmentation counts");st.json(d.get("segmentation",{}));st.write("Paired segmentation classes",d.get("paired_by_class",{}));st.write("Image dimensions",d.get("image_dimensions",{}));st.write("Mask pixel values",d.get("mask_unique_values",{}));st.write("Warnings",d.get("warnings",[]));st.write("Corrupt or dimension errors",len(d.get("corrupt_or_dimension_errors",[])));st.write("Exact duplicate groups",len(d.get("sha256_duplicate_groups",[])))
else:st.info("Dataset audit is not available. Download data and run `python scripts/audit_dataset.py`.")
