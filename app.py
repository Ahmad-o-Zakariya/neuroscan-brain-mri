"""NeuroScan multi-task inference app. Training is performed in the notebook only."""
from pathlib import Path
import io
import streamlit as st
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError

CLASSES=("glioma","meningioma","pituitary","non_tumorous")
CHECKPOINT=Path("artifacts/checkpoints/best_multitask_model.pth")

class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.GELU(),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.GELU())
    def forward(self,x): return self.net(x)

class MultiTaskModel(nn.Module):
    STAGE_IDS=(1,3,5,7); DIMS=(96,192,384,768)
    def __init__(self,architecture,input_size=224):
        super().__init__(); self.architecture=architecture;self.input_size=int(input_size)
        from torchvision.models import convnext_tiny,swin_t
        if architecture=="convnext_tiny": backbone=convnext_tiny(weights=None)
        elif architecture=="swin_tiny": backbone=swin_t(weights=None)
        else: raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
        self.encoder=backbone.features
        self.classifier=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,len(CLASSES)))
        self.proj=nn.ModuleList([nn.Conv2d(96,96,1),nn.Conv2d(192,128,1),nn.Conv2d(384,128,1),nn.Conv2d(768,128,1)])
        self.dec3=ConvBlock(256,128);self.dec2=ConvBlock(256,96);self.dec1=ConvBlock(192,64);self.mask_head=nn.Conv2d(64,1,1)
    def forward(self,x):
        skips=[]
        for i,layer in enumerate(self.encoder):
            x=layer(x)
            if i in self.STAGE_IDS:
                skips.append(x.permute(0,3,1,2).contiguous() if self.architecture=="swin_tiny" else x)
        f1,f2,f3,f4=skips; cls=self.classifier(F.adaptive_avg_pool2d(f4,1).flatten(1))
        d=F.interpolate(self.proj[3](f4),size=f3.shape[-2:],mode="bilinear",align_corners=False);d=self.dec3(torch.cat([d,self.proj[2](f3)],1))
        d=F.interpolate(d,size=f2.shape[-2:],mode="bilinear",align_corners=False);d=self.dec2(torch.cat([d,self.proj[1](f2)],1))
        d=F.interpolate(d,size=f1.shape[-2:],mode="bilinear",align_corners=False);d=self.dec1(torch.cat([d,self.proj[0](f1)],1))
        seg=F.interpolate(self.mask_head(d),size=(self.input_size,self.input_size),mode="bilinear",align_corners=False)
        return {"classification":cls,"segmentation":seg}

@st.cache_resource(show_spinner="Loading multi-task checkpoint...")
def load_model(checkpoint_path,mtime):
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    metadata=checkpoint.get("metadata",{});architecture=metadata.get("architecture");size=int(metadata.get("input_size",224))
    model=MultiTaskModel(architecture,size)
    model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval()
    return model,metadata

st.set_page_config(page_title="NeuroScan Multi-Task MRI",page_icon="🧠",layout="wide")
st.title("NeuroScan · Brain MRI Research")
st.caption("Shared-encoder classification and tumor-mask segmentation · research/education only")
if not CHECKPOINT.is_file():
    st.error(f"Multi-task checkpoint is not available: `{CHECKPOINT}`. Run the training and export cells in `train_brain_tumor_multitask.ipynb` first.")
    st.info("The existing EfficientNet-B0 baseline is preserved separately and is not treated as a multi-task model.")
else:
    try:
        model,meta=load_model(str(CHECKPOINT.resolve()),CHECKPOINT.stat().st_mtime_ns)
        st.sidebar.subheader("Loaded model")
        st.sidebar.write(f"Architecture: `{meta.get('architecture','unknown')}`")
        st.sidebar.write(f"Input: {meta.get('input_size','?')} × {meta.get('input_size','?')}")
        st.sidebar.write(f"Glioma threshold: {meta.get('glioma_threshold','not calibrated')}")
        st.sidebar.write(f"Checkpoint: `{CHECKPOINT}`")
        upload=st.file_uploader("Upload an MRI slice",type=["jpg","jpeg","png"])
        if upload:
            if upload.size>20*1024*1024: st.error("Image exceeds the 20 MB upload limit.")
            else:
                try:
                    image=Image.open(upload);image.verify();upload.seek(0);original=ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
                    if max(original.size)>8192 or original.width*original.height>32_000_000: raise ValueError("Image dimensions exceed the supported limit.")
                    col1,col2=st.columns(2);col1.image(original,caption=f"Original MRI · {original.width} × {original.height}",use_container_width=True)
                    if st.button("Predict",type="primary"):
                        size=int(meta.get("input_size",224));mean=meta.get("normalization",{}).get("mean",(.485,.456,.406));std=meta.get("normalization",{}).get("std",(.229,.224,.225))
                        resized=original.resize((size,size),Image.Resampling.BILINEAR);x=torch.from_numpy(np.asarray(resized,dtype=np.float32).copy()).permute(2,0,1)/255;x=(x-torch.tensor(mean)[:,None,None])/torch.tensor(std)[:,None,None]
                        with torch.inference_mode():
                            out=model(x.unsqueeze(0));prob=torch.softmax(out["classification"],1)[0].numpy();maskprob=torch.sigmoid(out["segmentation"])[0,0].numpy()
                        label_idx=int(prob.argmax());selected_threshold=meta.get("glioma_threshold")
                        threshold=float(selected_threshold) if selected_threshold is not None else float("nan")
                        if selected_threshold is not None:
                            label_idx=int(1+prob[1:].argmax())
                            if prob[0]>=threshold:label_idx=0
                        mask_thr=float(meta.get("mask_threshold",.5));binary=(maskprob>=mask_thr).astype(np.uint8)
                        mask=Image.fromarray(binary*255).resize(original.size,Image.Resampling.NEAREST)
                        overlay=original.convert("RGBA");red=Image.new("RGBA",original.size,(235,40,55,0));red.putalpha(mask.point(lambda v:int(v*.4)));overlay=Image.alpha_composite(overlay,red).convert("RGB")
                        col2.image(overlay,caption="Predicted segmentation overlay",use_container_width=True)
                        st.subheader("Classification")
                        st.metric("Predicted class",CLASSES[label_idx],f"Confidence {prob[label_idx]:.1%}")
                        st.write((f"Glioma threshold: {threshold:.3f} · glioma probability: {prob[0]:.1%}") if selected_threshold is not None else f"Glioma probability: {prob[0]:.1%} · no calibrated glioma threshold in checkpoint")
                        st.bar_chart({name:float(v) for name,v in zip(CLASSES,prob)})
                        st.subheader("Predicted tumor mask")
                        st.image(mask,caption="Predicted binary mask",clamp=True,width=320)
                        buf=io.BytesIO();mask.save(buf,format="PNG");st.download_button("Download predicted mask",buf.getvalue(),"predicted_tumor_mask.png","image/png")
                except (UnidentifiedImageError,OSError,ValueError) as exc: st.error(f"Invalid or unreadable image: {exc}")
                except Exception as exc: st.error(f"Inference failed: {exc}")
    except Exception as exc:
        st.error(f"Checkpoint is incompatible or could not be loaded: {exc}")
st.warning("Research/education only; not a medical device. This predicts from one 2D T1CE slice and cannot diagnose a patient or replace clinical review.")
