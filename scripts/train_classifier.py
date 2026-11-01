import argparse, json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd, torch
from torch.utils.data import DataLoader
from src.config import CLASSES, MEAN, STD
from src.data.audit import discover_root,audit_dataset
from src.data.dataset import ClassificationDataset, classification_records, write_splits
from src.models.networks import build_classifier
from src.training.common import seed_all, device_report, save_checkpoint

def main():
 p=argparse.ArgumentParser(); p.add_argument("--data-root",default=str(ROOT / "data/raw")); p.add_argument("--config",default=str(ROOT / "configs/default.yaml")); p.add_argument("--smoke-test",action="store_true"); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--size",type=int,default=224); p.add_argument("--workers",type=int,default=2); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
 root=discover_root(a.data_root); audit=audit_dataset(root,ROOT / "reports")
 if audit["classification"].get("integrity_errors"): raise SystemExit(f"Classification audit found unreadable images ({len(audit['classification']['integrity_errors'])}); inspect before training.")
 seed_all(a.seed); splits=write_splits(root,a.seed,ROOT / "data/splits")
 tr=pd.read_csv(splits/"classification_train.csv").to_dict("records"); va=pd.read_csv(splits/"classification_validation.csv").to_dict("records")
 if not tr: raise SystemExit("No classification training records found.")
 dev=device_report()["device"]; print(json.dumps(device_report())); model=build_classifier(pretrained=not a.smoke_test).to(dev)
 if a.smoke_test:
  ds=ClassificationDataset(tr[:2],a.size); x,y=next(iter(DataLoader(ds,batch_size=2))); z=model(x.to(dev)); loss=torch.nn.functional.cross_entropy(z,y.to(dev)); loss.backward(); print(f"smoke forward/backward ok: {tuple(z.shape)} loss={loss.item():.4f}"); return
 for n,pv in model.named_parameters(): pv.requires_grad="classifier" in n
 train=DataLoader(ClassificationDataset(tr,a.size,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev=="cuda")
 valid=DataLoader(ClassificationDataset(va,a.size),batch_size=a.batch_size,num_workers=a.workers)
 # Class-weighted loss: inverse-square-root frequency, normalized to mean 1.
 counts = pd.Series([row["label"] for row in tr]).value_counts()
 class_weights = torch.tensor(
  [(len(tr) / (len(CLASSES) * counts[c])) ** 0.5 for c in CLASSES],
  dtype=torch.float32, device=dev
 )
 print("Training class counts:", {c: int(counts[c]) for c in CLASSES})
 print("Class weights:", {c: round(float(class_weights[i]), 4) for i, c in enumerate(CLASSES)})

 opt=torch.optim.AdamW(filter(lambda x:x.requires_grad,model.parameters()),lr=1e-3,weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda",enabled=dev=="cuda"); best=0.; patience=0; history=[]
 for epoch in range(a.epochs):
  model.train(); losses=[]
  for x,y in train:
   x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=dev,enabled=dev=="cuda"): loss=torch.nn.functional.cross_entropy(model(x),y,weight=class_weights,label_smoothing=.03)
   scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(loss.item())
  model.eval(); good=total=0
  with torch.inference_mode():
   for x,y in valid:
    z=model(x.to(dev)); good+=(z.argmax(1).cpu()==y).sum().item(); total+=len(y)
  acc=good/max(1,total); history.append({"epoch":epoch+1,"train_loss":sum(losses)/len(losses),"val_accuracy":acc}); print(history[-1])
  if acc>best:
   best=acc; patience=0; save_checkpoint(ROOT / "artifacts/checkpoints/classifier_best.pth",model,{"architecture":"efficientnet_b0","classes":CLASSES,"input_size":a.size,"normalization":{"mean":MEAN,"std":STD},"seed":a.seed,"validation_accuracy":best,"device":dev,"tumour_threshold":.5,"mask_threshold":.5})
  else: patience+=1
  if epoch>=5:
   for n,pv in model.named_parameters():
    if n.startswith("features.7.") or n.startswith("features.8."): pv.requires_grad=True
   if epoch==5: opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
  if patience>=5: break
 (ROOT / "reports").mkdir(exist_ok=True); (ROOT / "reports/classifier_history.json").write_text(json.dumps(history,indent=2))
if __name__=="__main__": main()
