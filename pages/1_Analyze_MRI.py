import json
from pathlib import Path
import io
import streamlit as st
from PIL import Image,UnidentifiedImageError
from src.inference.pipeline import analyze_image,annotated_image,gradcam
st.set_page_config(page_title="Analyze MRI | NeuroScan",layout="wide");st.title("Analyze MRI")
upload=st.file_uploader("Upload JPG, JPEG, or PNG",type=["jpg","jpeg","png"]);view=st.selectbox("View (user supplied)",["Unknown","Axial","Coronal","Sagittal"]);opacity=st.slider("Overlay opacity",.1,.8,.4);show_cam=st.checkbox("Show optional classifier Grad-CAM")
if upload:
 if upload.size>20*1024*1024: st.error("File exceeds 20 MB.")
 else:
  try:
   im=Image.open(upload); im.verify(); upload.seek(0); im=Image.open(upload).convert("RGB")
   if not Path("artifacts/checkpoints/classifier_best.pth").exists(): st.error("Prediction disabled: train classifier checkpoint first.")
   else:
    with st.spinner("Running slice-level analysis..."): result,orig=analyze_image(im)
    a,b=st.columns(2);a.image(orig,caption=f"Input · {orig.width} × {orig.height}",use_container_width=True)
    if result.mask is not None:b.image(annotated_image(orig,result.mask,opacity,result.boxes,result.centroids),caption="Predicted mask overlay",use_container_width=True)
    else:b.info("Screening gate did not pass; localization was not run.")
    st.subheader("Model output");st.write(f"Predicted class: **{result.predicted_label}** · tumour screening probability: **{result.tumour_probability:.1%}** · user supplied view: {view}")
    st.bar_chart(result.probabilities);st.write(f"Predicted foreground: {result.foreground_percent:.3f}% of pixels")
    if show_cam:
     st.caption("Grad-CAM shows classifier attention and is not a segmentation mask.")
     st.image(gradcam(orig),caption="Classifier Grad-CAM",clamp=True,use_container_width=True)
    for warning in result.warnings:st.warning(warning)
    st.download_button("Download result JSON",json.dumps(result.to_dict(),indent=2),"analysis.json","application/json")
    if result.mask is not None:
     buf=io.BytesIO();annotated_image(orig,result.mask,opacity,result.boxes,result.centroids).save(buf,format="PNG");st.download_button("Download annotated PNG",buf.getvalue(),"annotated.png","image/png")
  except (UnidentifiedImageError,OSError,ValueError) as e:st.error(f"Could not read this image: {e}")
