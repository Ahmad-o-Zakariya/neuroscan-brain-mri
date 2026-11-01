import io,zipfile
import pandas as pd,streamlit as st
from PIL import Image
from src.inference.pipeline import analyze_image,annotated_image
st.set_page_config(page_title="Batch Analysis | NeuroScan",layout="wide");st.title("Batch analysis")
files=st.file_uploader("Choose MRI images or a ZIP of images",type=["jpg","jpeg","png","zip"],accept_multiple_files=True)
if files and st.button("Analyze batch"):
 items=[]
 for f in files:
  if f.name.lower().endswith(".zip"):
   if f.size>200*1024*1024:st.warning(f"Skipped archive larger than 200 MB: {f.name}");continue
   try:
    with zipfile.ZipFile(f) as z:
     for info in z.infolist()[:500]:
      name=info.filename
      if not name.startswith("/") and ".." not in name.split("/") and name.lower().endswith((".jpg",".jpeg",".png")):
       if info.file_size<=20*1024*1024:items.append((name,z.read(info)))
   except zipfile.BadZipFile:st.warning(f"Skipped corrupt archive {f.name}")
  elif f.size<=20*1024*1024:items.append((f.name,f.getvalue()))
 rows=[];out=io.BytesIO();done=st.progress(0)
 with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
  for i,(name,data) in enumerate(items):
   try:
    im=Image.open(io.BytesIO(data)).convert("RGB");r,orig=analyze_image(im);rows.append({"filename":name,"predicted_label":r.predicted_label,"tumour_probability":r.tumour_probability,"gate_passed":r.gate_passed,"foreground_percent":r.foreground_percent,"warning":"; ".join(r.warnings)})
    if r.mask is not None:
     b=io.BytesIO();annotated_image(orig,r.mask,boxes=r.boxes,centroids=r.centroids).save(b,format="PNG");z.writestr(f"{i}_annotated.png",b.getvalue())
   except Exception as e:rows.append({"filename":name,"error":str(e)})
   done.progress((i+1)/max(1,len(items)))
 df=pd.DataFrame(rows);st.dataframe(df,use_container_width=True);st.download_button("Download summary CSV",df.to_csv(index=False),"batch_summary.csv","text/csv");st.download_button("Download annotations ZIP",out.getvalue(),"batch_annotations.zip","application/zip")
