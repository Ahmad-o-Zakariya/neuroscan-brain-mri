"""Evaluate frozen models; thresholds are fit only on generated validation CSVs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from PIL import Image
from src.config import CLASSES,FOLDER_LABELS
from src.data.audit import discover_root
from src.data.dataset import classification_records,segmentation_records,ClassificationDataset,SegmentationDataset,load_generated_split
from src.models.networks import build_classifier,build_segmenter
from src.training.common import load_checkpoint
from src.training.calibration import select_tumour_threshold,select_mask_threshold

def _records(root,task,split,split_dir):
    if split=="validation":
        try:return load_generated_split(split_dir,task,"validation")
        except FileNotFoundError as e:raise FileNotFoundError(f"{e}; run split generation first.") from e
    return classification_records(root,"test") if task=="classification" else segmentation_records(root,"test")

def _plot_classifier(y,pred,prob,out):
    cm=confusion_matrix(y,pred,labels=range(4)); fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm,cmap="Blues");fig.colorbar(im,ax=ax)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,xlabel="Predicted",ylabel="True",title="Classification confusion matrix")
    for i in range(4):
        for j in range(4): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="black")
    fig.tight_layout();fig.savefig(out/"classifier_confusion_matrix.png",dpi=150);plt.close(fig)
    ybin=label_binarize(y,classes=range(4));fig,ax=plt.subplots(figsize=(6,5));valid=[]
    for i,name in enumerate(CLASSES):
        if len(np.unique(ybin[:,i]))<2: continue
        fpr,tpr,_=roc_curve(ybin[:,i],prob[:,i]);ax.plot(fpr,tpr,label=f"{name} (AUC {auc(fpr,tpr):.3f})");valid.append(i)
    if valid: ax.plot([0,1],[0,1],"k--",lw=1);ax.legend()
    ax.set(xlabel="False positive rate",ylabel="True positive rate",title="One-vs-rest ROC");fig.tight_layout();fig.savefig(out/"classifier_roc.png",dpi=150);plt.close(fig)

def _plot_segmentation(dice,iou,out):
    fig,axes=plt.subplots(1,2,figsize=(9,4));axes[0].hist(dice,bins=20,range=(0,1));axes[0].set(title="Per-image Dice",xlabel="Dice",ylabel="Images");axes[1].hist(iou,bins=20,range=(0,1));axes[1].set(title="Per-image IoU",xlabel="IoU",ylabel="Images");fig.tight_layout();fig.savefig(out/"segmenter_dice_iou.png",dpi=150);plt.close(fig)

def _evaluate_gated_cases(classification_rows,probabilities,segmentation_rows,threshold,segmenter,device,input_size,mask_threshold):
    by_stem={Path(r["image"]).stem.lower():r for r in segmentation_rows};gate_preds=[];paired_dice=[];not_run=run=no_pair=0
    for i,row in enumerate(classification_rows):
        label=FOLDER_LABELS.get(Path(row["path"]).parent.name.lower(),Path(row["path"]).parent.name.lower())
        if label not in CLASSES[:3]: continue
        passed=float(probabilities[i,:3].sum())>=threshold;gate_preds.append(passed);segrow=by_stem.get(Path(row["path"]).stem.lower())
        if not passed:
            not_run+=1
            if segrow is not None: paired_dice.append(0.)
        elif segrow is None:no_pair+=1
        else:
            run+=1;item=SegmentationDataset([segrow],input_size)[0][0].unsqueeze(0).to(device)
            with torch.inference_mode():pm=torch.sigmoid(segmenter(item))[0,0].cpu().numpy()
            gt=np.asarray(Image.open(segrow["mask"]).convert("L").resize((pm.shape[1],pm.shape[0]),Image.Resampling.NEAREST))>0;pb=pm>=mask_threshold;inter=np.logical_and(pb,gt).sum();den=pb.sum()+gt.sum();paired_dice.append(float(1 if den==0 else 2*inter/den))
    return {"split":"test","n_tumour_classification_cases":len(gate_preds),"tumour_screening_sensitivity":float(np.mean(gate_preds)) if gate_preds else None,"localization_not_run_after_gate":not_run,"localization_run_and_scored":run,"gate_passed_without_segmentation_pair":no_pair,"end_to_end_dice_mean_including_gate_false_negatives":float(np.mean(paired_dice)) if paired_dice else None,"n_paired_cases_scored":len(paired_dice),"threshold_source":"validation calibration"}

def _load_classifier(path,device):
    ck=torch.load(path,map_location=device,weights_only=True);md=ck["metadata"];model=build_classifier(pretrained=False);load_checkpoint(path,model,device,"efficientnet_b0");model.to(device).eval();return model,md

def _load_segmenter(path,device,architecture=None):
    ck=torch.load(path,map_location=device,weights_only=True);md=ck["metadata"];arch=architecture or md.get("architecture","attention_unet")
    if arch!=md.get("architecture"): raise ValueError(f"Requested {arch}, checkpoint metadata says {md.get('architecture')}.")
    model=build_segmenter(arch,md.get("base_channels",16));load_checkpoint(path,model,device,arch);model.to(device).eval();return model,md

def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["classifier","segmenter","all"],default="all");p.add_argument("--split",choices=["test","validation"],default="test");p.add_argument("--data-root",default=str(ROOT / "data/raw"));p.add_argument("--split-dir",default=str(ROOT / "data/splits"));p.add_argument("--architecture",choices=["unet","attention_unet"],default=None);a=p.parse_args()
    root=discover_root(a.data_root);device="cuda" if torch.cuda.is_available() else "cpu";out=ROOT/"reports";figdir=out/"figures";out.mkdir(exist_ok=True);figdir.mkdir(exist_ok=True);cal_path=out/"calibration.json";cal=json.loads(cal_path.read_text()) if cal_path.exists() else {}; calibration_changed=False
    classifier=None;cmeta={};crows=[];probs=None;y=None;pred=None
    if a.task in ("classifier","all"):
        path=ROOT/"artifacts/checkpoints/classifier_best.pth"
        if not path.exists(): raise SystemExit(f"Missing {path}")
        classifier,cmeta=_load_classifier(path,device);crows=_records(root,"classification",a.split,a.split_dir)
        if not crows: raise SystemExit(f"No classification records available for split={a.split}.")
        dsrows=[dict(r,path=str(r.get("path","")),label=FOLDER_LABELS.get(str(r.get("label","")).lower(),str(r.get("label","")))) for r in crows]
        dl=DataLoader(ClassificationDataset(dsrows,cmeta.get("input_size",224)),batch_size=8);ys=[];ps=[]
        with torch.inference_mode():
            for x,b in dl: ps.extend(torch.softmax(classifier(x.to(device)),1).cpu().numpy());ys.extend(b.numpy())
        y=np.asarray(ys);probs=np.asarray(ps);pred=probs.argmax(1)
        metrics={"split":a.split,"n":len(y),"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"per_class":classification_report(y,pred,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(y,pred,labels=range(4)).tolist()}
        try: metrics["macro_ovr_auc"]=roc_auc_score(y,probs,multi_class="ovr",average="macro",labels=range(4))
        except ValueError: metrics["macro_ovr_auc"]=None
        metrics["brier_score"]=float(np.mean(np.sum((probs-np.eye(4)[y])**2,axis=1)));(out/"classifier_metrics.json").write_text(json.dumps(metrics,indent=2));_plot_classifier(y,pred,probs,figdir)
        if a.split=="validation":
            cal["tumour_gate"]=select_tumour_threshold(probs[:,:3].sum(axis=1),y<3);cal["source_split"]="validation";calibration_changed=True
        print(json.dumps(metrics,indent=2))
    architecture=a.architecture;segmenter=None;smeta={};srows=[];s_dice=[];s_iou=[];probs_seg=[];masks_seg=[]
    if a.task in ("segmenter","all"):
        if architecture is None:
            candidates=[x for x in ("attention_unet","unet") if Path(f"artifacts/checkpoints/segmenter_{x}_best.pth").exists()]
            if not candidates: raise SystemExit("No segmenter checkpoint found (expected segmenter_unet_best.pth or segmenter_attention_unet_best.pth).")
            architecture=candidates[0]
        path=ROOT/f"artifacts/checkpoints/segmenter_{architecture}_best.pth"
        if not path.exists(): raise SystemExit(f"Missing {path}")
        segmenter,smeta=_load_segmenter(path,device,architecture);srows=_records(root,"segmentation",a.split,a.split_dir)
        if not srows: raise SystemExit(f"No segmentation records available for split={a.split}.")
        dl=DataLoader(SegmentationDataset(srows,smeta.get("input_size",256)),batch_size=4)
        with torch.inference_mode():
            for x,target in dl:
                prob=torch.sigmoid(segmenter(x.to(device))).cpu().numpy();probs_seg.extend(prob[:,0]);masks_seg.extend(target[:,0].numpy())
        probs_seg=np.asarray(probs_seg);masks_seg=np.asarray(masks_seg)
        if a.split=="validation":
            cal.setdefault("segmentation_mask",{})[architecture]=select_mask_threshold(probs_seg,masks_seg);cal["source_split"]="validation";calibration_changed=True
        threshold=cal.get("segmentation_mask",{}).get(architecture,{}).get("threshold",smeta.get("mask_threshold",.5));tp=fp=fn=tn=0
        for prob,truth in zip(probs_seg,masks_seg):
            binary=prob>=threshold;truth=truth>=.5;inter=np.logical_and(binary,truth).sum();den=binary.sum()+truth.sum();union=np.logical_or(binary,truth).sum();s_dice.append(float(1. if den==0 else 2*inter/den));s_iou.append(float(1. if union==0 else inter/union));tp+=int(np.logical_and(binary,truth).sum());fp+=int(np.logical_and(binary,~truth).sum());fn+=int(np.logical_and(~binary,truth).sum());tn+=int(np.logical_and(~binary,~truth).sum())
        smetrics={"split":a.split,"architecture":architecture,"threshold":float(threshold),"n_images":len(srows),"dice_per_image_mean":float(np.mean(s_dice)),"iou_per_image_mean":float(np.mean(s_iou)),"pixel_precision":tp/max(1,tp+fp),"pixel_sensitivity":tp/max(1,tp+fn),"pixel_specificity":tn/max(1,tn+fp)};(out/"segmenter_metrics.json").write_text(json.dumps(smetrics,indent=2))
        _plot_segmentation(s_dice,s_iou,figdir);print(json.dumps(smetrics,indent=2))
    if a.task=="all" and a.split=="test" and classifier is not None and segmenter is not None:
        if not cal.get("tumour_gate"): print("End-to-end evaluation skipped: run classifier validation calibration first.")
        else:
            gate=float(cal["tumour_gate"]["threshold"]);test_cls=crows;test_seg=segmentation_records(root,"test");maskthr=float(cal.get("segmentation_mask",{}).get(architecture,{}).get("threshold",smeta.get("mask_threshold",.5)))
            e2e=_evaluate_gated_cases(test_cls,probs,test_seg,gate,segmenter,device,smeta.get("input_size",256),maskthr);(out/"end_to_end_metrics.json").write_text(json.dumps(e2e,indent=2));print(json.dumps(e2e,indent=2))
    if calibration_changed: cal_path.write_text(json.dumps(cal,indent=2))

if __name__=="__main__": main()
