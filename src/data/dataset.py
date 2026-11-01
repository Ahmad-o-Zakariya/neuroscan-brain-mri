from pathlib import Path
import random
import hashlib
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from src.config import CLASSES, FOLDER_LABELS, MEAN, STD

def classification_records(root, split):
    base=Path(root)/"classification_task"/split; rows=[]
    for folder in sorted(base.iterdir()):
        label=FOLDER_LABELS.get(folder.name.lower())
        if label:
            rows += [{"path":str(f),"label":label,"split":split} for f in folder.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}]
    return rows

def load_generated_split(split_dir,task,split):
    if split not in {"train","validation"} or task not in {"classification","segmentation"}: raise ValueError("Only generated train/validation splits are supported.")
    path=Path(split_dir)/f"{task}_{split}.csv"
    if not path.exists(): raise FileNotFoundError(f"Generated split CSV missing: {path}")
    frame=pd.read_csv(path).fillna("")
    required={"path","label"} if task=="classification" else {"image","mask"}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f"Generated split CSV {path} lacks required columns: {sorted(missing)}")
    return frame.to_dict("records")

def segmentation_records(root, split):
    base=Path(root)/"segmentation_task"/split; imgs={p.stem.lower():p for p in (base/"images").glob("*")}; masks={p.stem.lower():p for p in (base/"masks").glob("*")}
    rows=[]
    for k in sorted(imgs.keys()&masks.keys()):
        if imgs[k].suffix.lower() not in {".jpg",".jpeg",".png"} or masks[k].suffix.lower() not in {".png",".jpg",".jpeg"}: continue
        stem=imgs[k].stem.lower(); label=next((x for x in ("glioma","meningioma","pituitary") if stem.startswith(x) or f"_{x}_" in f"_{stem}_"),"unknown")
        rows.append({"image":str(imgs[k]),"mask":str(masks[k]),"split":split,"label":label})
    return rows

def _record_image(row):
    """Return the source image path for either task's record schema."""
    return row.get("image") or row.get("path")

def _digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def _split_duplicate_groups(rows, rng):
    """Split by exact image-content groups, preserving class stratification."""
    by_label={}
    for row in rows:
        path=_record_image(row)
        if not path: raise ValueError(f"Split record has neither image nor path: {row}")
        key=_digest(path); label=row.get("label","unknown")
        by_label.setdefault(label,{}).setdefault(key,[]).append(row)
    train=[]; val=[]; membership={}
    # A content hash seen under multiple classes is kept in one deterministic
    # stratum so copies can never cross the boundary.
    hash_labels={}
    for label,groups in by_label.items():
        for key in groups: hash_labels.setdefault(key,set()).add(label)
    merged={}
    for label,groups in by_label.items():
        for key,items in groups.items():
            stratum="|".join(sorted(hash_labels[key])); merged.setdefault(stratum,{}).setdefault(key,[]).extend(items)
    for stratum,groups in merged.items():
        keys=list(groups); rng.shuffle(keys)
        # Keep at least one unique image group in train for every stratum.
        n=max(1,round(len(keys)*.2)) if len(keys)>1 else 0
        val_keys=set(keys[:n])
        for key,items in groups.items():
            split="validation" if key in val_keys else "train"; membership[key]=split
            (val if split=="validation" else train).extend(items)
    return train,val,membership

def check_split_leakage(root, split_dir="data/splits"):
    """Check exact-content overlap among generated train/validation and official test records."""
    root=Path(root); split_dir=Path(split_dir); findings={}
    for task,record_fn in (("classification",classification_records),("segmentation",segmentation_records)):
        tr=pd.read_csv(split_dir/f"{task}_train.csv").to_dict("records")
        va=pd.read_csv(split_dir/f"{task}_validation.csv").to_dict("records")
        test=record_fn(root,"test"); digest=lambda rows:{_digest(_record_image(r)) for r in rows}
        train_hashes=digest(tr); val_hashes=digest(va); test_hashes=digest(test)
        findings[task]={"train_validation_overlap":sorted(train_hashes&val_hashes),"train_test_overlap":sorted(train_hashes&test_hashes),"validation_test_overlap":sorted(val_hashes&test_hashes)}
        if any(findings[task].values()): raise ValueError(f"Exact duplicate leakage detected for {task}: {findings[task]}")
    cls={};seg={}
    for task,record_fn,target in (("classification",classification_records,cls),("segmentation",segmentation_records,seg)):
        for split in ("train","validation"):
            rows=pd.read_csv(split_dir/f"{task}_{split}.csv").to_dict("records")
            target.update({_digest(_record_image(r)):split for r in rows})
        target.update({_digest(_record_image(r)):"test" for r in record_fn(root,"test")})
    cross=[h for h in cls.keys()&seg.keys() if cls[h]!=seg[h]]
    findings["cross_task_split_overlap"]=sorted(cross)
    if cross: raise ValueError(f"Cross-task exact duplicates span different train/validation/test splits: {cross[:10]}")
    return findings

def write_splits(root, seed=42, split_dir="data/splits"):
    out=Path(split_dir); out.mkdir(parents=True,exist_ok=True); rng=random.Random(seed)
    shared_assignments={}
    for task,record_fn in (("classification",classification_records),("segmentation",segmentation_records)):
        rows=record_fn(root,"train")
        # Exclude any training records whose exact image content appears in
        # the official test set. Keep the raw dataset and official test split intact.
        test_hashes = {_digest(_record_image(row)) for row in record_fn(root,"test")}
        rows = [row for row in rows if _digest(_record_image(row)) not in test_hashes]
        if task=="classification":
            for row in rows: row.setdefault("label",Path(row["path"]).parent.name)
        train,val,membership=_split_duplicate_groups(rows,rng)
        if task=="classification": shared_assignments=membership
        else:
            forced={h:shared_assignments[h] for h in membership if h in shared_assignments}
            if forced:
                train=[];val=[]
                for row in rows:
                    split=forced.get(_digest(_record_image(row)),membership[_digest(_record_image(row))])
                    (val if split=="validation" else train).append(row)
        columns=list(rows[0]) if rows else (["path","label","split"] if task=="classification" else ["image","mask","split","label"])
        pd.DataFrame(train,columns=columns).to_csv(out/f"{task}_train.csv",index=False); pd.DataFrame(val,columns=columns).to_csv(out/f"{task}_validation.csv",index=False)
    check_split_leakage(root,out)
    return out

class ClassificationDataset(Dataset):
    def __init__(self, rows, size=224, augment=False): self.rows=rows; self.size=size; self.augment=augment; self.norm=transforms.Normalize(MEAN,STD)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; im=Image.open(row["path"]).convert("L").resize((self.size,self.size),Image.Resampling.BILINEAR)
        if self.augment:
            if random.random()<.5: im=TF.rotate(im,random.uniform(-8,8))
            if random.random()<.5: im=ImageEnhance.Brightness(im).enhance(random.uniform(.9,1.1))
        x=transforms.ToTensor()(im).repeat(3,1,1); x=self.norm(x)
        if self.augment: x=(x+torch.randn_like(x)*.015).clamp(-3,3)
        return x,CLASSES.index(row["label"])

class SegmentationDataset(Dataset):
    def __init__(self,rows,size=256,augment=False): self.rows=rows; self.size=size; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; im=Image.open(r["image"]).convert("L").resize((self.size,self.size),Image.Resampling.BILINEAR); ma=Image.open(r["mask"]).convert("L").resize((self.size,self.size),Image.Resampling.NEAREST)
        if self.augment and random.random()<.5: im,ma=TF.hflip(im),TF.hflip(ma)
        x=transforms.ToTensor()(im); y=(transforms.ToTensor()(ma)>0.5).float()
        return x,y
