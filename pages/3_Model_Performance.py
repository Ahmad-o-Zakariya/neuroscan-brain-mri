import json
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="Performance | NeuroScan",layout="wide");st.title("Model performance")
for name in ("classifier_metrics.json","segmenter_metrics.json","end_to_end_metrics.json","calibration.json","classifier_history.json","segmenter_history.json"):
 p=Path("reports")/name
 if p.exists():
  st.subheader(name.replace("_"," ").replace(".json",""));d=json.loads(p.read_text());st.json(d)
  stems={"classifier_metrics.json":["classifier_confusion_matrix.png","classifier_roc.png"],"segmenter_metrics.json":["segmenter_dice_iou.png"]}
  for img in stems.get(name,[]):
   fig=Path("reports/figures")/img
   if fig.exists():st.image(str(fig),caption=img.replace("_"," ").replace(".png",""))
if not any((Path("reports")/n).exists() for n in ("classifier_metrics.json","segmenter_metrics.json")):st.info("No evaluation metrics are available yet. Train models and run evaluation; metrics are not estimated or fabricated here.")
st.caption("Training histories shown separately from official test evaluation. Do not use official test data for model selection.")
