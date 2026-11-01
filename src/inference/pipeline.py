"""Canonical two-stage inference and pixel-coordinate localization."""
import time
import json
from functools import lru_cache
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torchvision import transforms
from src.config import CLASSES,MEAN,STD
from src.models.networks import build_classifier,build_segmenter
from src.training.common import load_checkpoint

@lru_cache(maxsize=4)
def _classifier(path,mtime,device):
 ck=torch.load(path,map_location=device,weights_only=True);meta=ck.get("metadata",{});model=build_classifier(pretrained=False);load_checkpoint(path,model,device,"efficientnet_b0");model.to(device).eval();return model,meta

@lru_cache(maxsize=4)
def _segmenter(path,mtime,device,architecture,channels):
 model=build_segmenter(architecture,channels);load_checkpoint(path,model,device,architecture);model.to(device).eval();return model

@dataclass
class AnalysisResult:
    predicted_label:str; probabilities:dict; tumour_probability:float; gate_passed:bool; mask:object; probability_map:object; boxes:list; centroids:list; foreground_percent:float; processing_seconds:float; warnings:list
    def to_dict(self):
        d=asdict(self); d["mask"]=None if self.mask is None else "included as PNG artifact";d["probability_map"]=None if self.probability_map is None else "available in memory";return d

def _components(mask):
    try:
        import cv2
        n,lab,stats,cents=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8); boxes=[];cent=[]
        for i in range(1,n):
            x,y,w,h,area=map(int,stats[i]);boxes.append({"x":x,"y":y,"width":w,"height":h,"area_pixels":area});cent.append({"x":float(cents[i][0]),"y":float(cents[i][1])})
        return boxes,cent
    except ImportError:
        ys,xs=np.where(mask); return ([{"x":int(xs.min()),"y":int(ys.min()),"width":int(xs.max()-xs.min()+1),"height":int(ys.max()-ys.min()+1),"area_pixels":int(mask.sum())}] if len(xs) else []),([{"x":float(xs.mean()),"y":float(ys.mean())}] if len(xs) else [])

def analyze_image(image,classifier_path="artifacts/checkpoints/classifier_best.pth",segmenter_path="artifacts/checkpoints/segmenter_attention_unet_best.pth",tumour_threshold=None,mask_threshold=None,device=None,calibration_path="reports/calibration.json"):
 t=time.perf_counter()
 if isinstance(image,(str,Path)): image=Image.open(image)
 if image.width<16 or image.height<16 or image.width>4096 or image.height>4096 or image.width*image.height>16_000_000: raise ValueError("Image dimensions must be at least 16×16 and no larger than 4096×4096 or 16 megapixels.")
 original=image.convert("RGB"); grey=original.convert("L"); warnings=[]; device=device or ("cuda" if torch.cuda.is_available() else "cpu")
 if not Path(classifier_path).exists():raise FileNotFoundError(f"Classifier checkpoint missing: {classifier_path}")
 classifier_path=str(Path(classifier_path).resolve());model,meta=_classifier(classifier_path,Path(classifier_path).stat().st_mtime,device);size=int(meta.get("input_size",224))
 prep=transforms.Compose([transforms.Resize((size,size),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor(),transforms.Lambda(lambda x:x.repeat(3,1,1)),transforms.Normalize(tuple(meta.get("normalization",{}).get("mean",MEAN)),tuple(meta.get("normalization",{}).get("std",STD)))])
 with torch.inference_mode(): probs=torch.softmax(model(prep(grey).unsqueeze(0).to(device)),1)[0].cpu().numpy()
 calibration={}
 if Path(calibration_path).exists():
  try: calibration=json.loads(Path(calibration_path).read_text())
  except (OSError,ValueError): warnings.append("Calibration file could not be read; checkpoint defaults are being used.")
 pmap=dict(zip(CLASSES,map(float,probs)));pt=float(sum(pmap[k] for k in CLASSES[:3])); calibrated_gate=calibration.get("tumour_gate",{}).get("threshold");gate=float(tumour_threshold if tumour_threshold is not None else calibrated_gate if calibrated_gate is not None else meta.get("tumour_threshold",.5)); passed=pt>=gate; mask=None;prob=None;boxes=[];cent=[];fraction=0.
 if passed:
  if not Path(segmenter_path).exists(): warnings.append("Screening threshold passed, but segmenter checkpoint is unavailable.")
  else:
   segmenter_path=str(Path(segmenter_path).resolve());sck=torch.load(segmenter_path,map_location=device,weights_only=True);sm=sck.get("metadata",{});sz=int(sm.get("input_size",256));architecture=sm.get("architecture","attention_unet");seg=_segmenter(segmenter_path,Path(segmenter_path).stat().st_mtime,device,architecture,int(sm.get("base_channels",16)));x=transforms.ToTensor()(grey.resize((sz,sz),Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
   with torch.inference_mode(): prob=torch.sigmoid(seg(x))[0,0].cpu().numpy()
   calibrated_masks=calibration.get("segmentation_mask",{}); calibrated_masks=calibrated_masks.get(architecture,{}) if isinstance(calibrated_masks,dict) else {}; calibrated_mask=calibrated_masks.get("threshold")
   thr=float(mask_threshold if mask_threshold is not None else calibrated_mask if calibrated_mask is not None else sm.get("mask_threshold",.5)); low=(prob>=thr).astype(np.uint8); mask=np.asarray(Image.fromarray(low*255).resize(original.size,Image.Resampling.NEAREST))>0; boxes,cent=_components(mask);fraction=float(mask.mean()*100)
 else: warnings.append("Model did not pass its tumour-screening threshold; this does not rule out a tumour.")
 if float(np.max(probs))<.5: warnings.append("Classifier output is low confidence (maximum class probability below 0.50).")
 return AnalysisResult(CLASSES[int(np.argmax(probs))],pmap,pt,passed,mask,prob,boxes,cent,fraction,time.perf_counter()-t,warnings),original

def gradcam(image,classifier_path="artifacts/checkpoints/classifier_best.pth",device=None):
 """Return a classifier Grad-CAM heatmap; it is not a segmentation mask."""
 if isinstance(image,(str,Path)): image=Image.open(image)
 grey=image.convert("L");device=device or ("cuda" if torch.cuda.is_available() else "cpu");path=str(Path(classifier_path).resolve())
 if not Path(path).exists():raise FileNotFoundError(f"Classifier checkpoint missing: {path}")
 model,meta=_classifier(path,Path(path).stat().st_mtime,device);size=int(meta.get("input_size",224));prep=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Lambda(lambda x:x.repeat(3,1,1)),transforms.Normalize(tuple(meta.get("normalization",{}).get("mean",MEAN)),tuple(meta.get("normalization",{}).get("std",STD)))])
 features={}
 def capture(_module,_inputs,output): output.retain_grad();features["map"]=output
 handle=model.features[-1].register_forward_hook(capture);x=prep(grey).unsqueeze(0).to(device).requires_grad_(True)
 try:
  model.zero_grad(set_to_none=True);logits=model(x);index=int(logits.argmax(1).item());logits[0,index].backward();act=features["map"];grad=act.grad
  if grad is None:raise RuntimeError("Grad-CAM gradients unavailable.")
  weights=grad.mean((2,3),keepdim=True);heat=torch.relu((weights*act).sum(1,keepdim=True));heat=torch.nn.functional.interpolate(heat,size=grey.size[::-1],mode="bilinear",align_corners=False)[0,0].detach().cpu().numpy();heat=heat/(float(heat.max())+1e-8);return heat
 finally:handle.remove();model.zero_grad(set_to_none=True)

def annotated_image(original,mask,opacity=.4,boxes=None,centroids=None):
 out=original.convert("RGBA")
 if mask is not None:
  m=Image.fromarray((mask.astype(np.uint8)*255)).convert("L"); red=Image.new("RGBA",out.size,(230,57,70,0));red.putalpha(m.point(lambda v:int(v*opacity)));out=Image.alpha_composite(out,red);draw=ImageDraw.Draw(out)
  try:
   import cv2
   contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
    points=[tuple(map(int,p[0])) for p in contour]
    if len(points)>1:draw.line(points+[points[0]],fill=(255,255,255,255),width=2)
  except ImportError: pass
  for b in boxes or []:draw.rectangle((b["x"],b["y"],b["x"]+b["width"],b["y"]+b["height"]),outline=(255,220,0,255),width=2)
  for c in centroids or []:draw.ellipse((c["x"]-3,c["y"]-3,c["x"]+3,c["y"]+3),fill=(0,255,255,255))
 return out.convert("RGB")
