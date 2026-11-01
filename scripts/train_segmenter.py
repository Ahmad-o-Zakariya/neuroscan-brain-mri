import argparse,json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd,torch
from torch.utils.data import DataLoader
from src.data.audit import discover_root,audit_dataset
from src.data.dataset import SegmentationDataset,write_splits
from src.models.networks import build_segmenter
from src.training.common import seed_all,device_report,save_checkpoint
from src.training.metrics import bce_dice_loss,dice_score

def main():
 p=argparse.ArgumentParser();p.add_argument("--data-root",default=str(ROOT / "data/raw"));p.add_argument("--config",default=str(ROOT / "configs/default.yaml"));p.add_argument("--model",choices=["unet","attention_unet"],default="attention_unet");p.add_argument("--smoke-test",action="store_true");p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--size",type=int,default=256);p.add_argument("--base-channels",type=int,default=16);p.add_argument("--workers",type=int,default=2);p.add_argument("--seed",type=int,default=42);p.add_argument("--accumulation",type=int,default=1);a=p.parse_args()
 root=discover_root(a.data_root); aud=audit_dataset(root,ROOT / "reports")
 if not aud["segmentation"]["train"]["matched_pairs"]: raise SystemExit("No matched train pairs; refusing to train.")
 if aud["corrupt_or_dimension_errors"]: raise SystemExit(f"Audit found corrupt files or image-mask dimension mismatches ({len(aud['corrupt_or_dimension_errors'])}); repair/inspect before training.")
 seed_all(a.seed);sp=write_splits(root,a.seed,ROOT / "data/splits");tr=pd.read_csv(sp/"segmentation_train.csv").to_dict("records");va=pd.read_csv(sp/"segmentation_validation.csv").to_dict("records"); dev=device_report()["device"];print(json.dumps(device_report()));m=build_segmenter(a.model,a.base_channels).to(dev)
 if a.smoke_test:
  x,y=next(iter(DataLoader(SegmentationDataset(tr[:2],a.size),batch_size=2))); z=m(x.to(dev)); loss=bce_dice_loss(z,y.to(dev));loss.backward();print(f"smoke forward/backward ok: {tuple(z.shape)} loss={loss.item():.4f}");return
 tl=DataLoader(SegmentationDataset(tr,a.size,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev=="cuda");vl=DataLoader(SegmentationDataset(va,a.size),batch_size=a.batch_size,num_workers=a.workers);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=2);sc=torch.amp.GradScaler("cuda",enabled=dev=="cuda");best=-1;wait=0;hist=[]
 for ep in range(a.epochs):
  m.train();ls=[];opt.zero_grad(set_to_none=True)
  for j,(x,y) in enumerate(tl):
   x,y=x.to(dev),y.to(dev)
   with torch.autocast(device_type=dev,enabled=dev=="cuda"): loss=bce_dice_loss(m(x),y)/a.accumulation
   sc.scale(loss).backward();
   if (j+1)%a.accumulation==0 or j+1==len(tl):sc.step(opt);sc.update();opt.zero_grad(set_to_none=True)
   ls.append(loss.item()*a.accumulation)
  m.eval();ds=[]
  with torch.inference_mode():
   for x,y in vl:ds.append(dice_score(m(x.to(dev)),y.to(dev)).item())
  vd=sum(ds)/max(1,len(ds));sch.step(vd);hist.append({"epoch":ep+1,"train_loss":sum(ls)/max(1,len(ls)),"val_dice_per_batch_mean":vd});print(hist[-1])
  if vd>best:
   best=vd;wait=0;save_checkpoint(ROOT / f"artifacts/checkpoints/segmenter_{a.model}_best.pth",m,{"architecture":a.model,"input_size":a.size,"base_channels":a.base_channels,"seed":a.seed,"mask_threshold":.5,"validation_dice_batch_mean":best,"device":dev,"positive_only":not aud["non_tumorous_masks_present"]})
  else:wait+=1
  if wait>=6:break
 (ROOT / "reports").mkdir(exist_ok=True);(ROOT / "reports/segmenter_history.json").write_text(json.dumps(hist,indent=2))
if __name__=="__main__":main()
