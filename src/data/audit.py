"""Dataset discovery and transparent structural/data-quality audit."""
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from src.config import FOLDER_LABELS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def discover_root(path):
    p = Path(path).expanduser().resolve()
    candidates = [p] + [x for x in p.iterdir() if x.is_dir()] if p.is_dir() else []
    for c in candidates:
        if (c / "classification_task").is_dir() and (c / "segmentation_task").is_dir():
            return c
    raise ValueError(f"Could not find classification_task/ and segmentation_task/ under {p}. Check the download path.")

def _files(p):
    return sorted(x for x in p.rglob("*") if x.is_file()) if p.exists() else []

def _manifest_labels(root):
    labels = {}
    for path in (root / "manifest.json", root / "segmentation_task/manifest.json"):
        if path.exists():
            try:
                data=json.loads(path.read_text(encoding="utf8")); rows=data if isinstance(data,list) else data.get("records",data.get("data",[])) if isinstance(data,dict) else []
                for row in rows:
                    if not isinstance(row,dict): continue
                    fname=next((row[k] for k in row if k.lower() in {"filename","image","image_name","path","file"} and row[k]),"")
                    label=next((row[k] for k in row if ("class" in k.lower() or "label" in k.lower()) and row[k]),"")
                    if fname and label: labels[Path(fname).name]=str(label).lower()
            except (ValueError,OSError): pass
    for path in (root / "manifest.csv", root / "segmentation_task/manifest.csv"):
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    fname = next((row[k] for k in row if k.lower() in {"filename", "image", "image_name", "path", "file"} and row[k]), "")
                    label = next((row[k] for k in row if ("class" in k.lower() or "label" in k.lower()) and row[k]), "")
                    if fname and label: labels[Path(fname).name] = str(label).lower()
    return labels

def audit_dataset(data_root, report_dir="reports"):
    root = discover_root(data_root); report_dir = Path(report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    audit = {"root": str(root), "classification": {}, "segmentation": {}, "warnings": [], "tree": []}
    for rel in ("classification_task", "segmentation_task"):
        audit["tree"].append({"path": rel, "directories": sorted(str(x.relative_to(root)) for x in (root / rel).rglob("*") if x.is_dir())[:100]})
    malformed=[]; classification_errors=[]; empty=[]; size_counts=Counter(); mask_values=Counter(); unexpected=[]
    for split in ("train", "test"):
        counts = Counter()
        for folder in (root / "classification_task" / split).iterdir() if (root / "classification_task" / split).is_dir() else []:
            if folder.is_dir(): counts[FOLDER_LABELS.get(folder.name.lower(), folder.name)] = sum(f.suffix.lower() in IMAGE_EXTS for f in _files(folder))
        audit["classification"][split] = dict(counts)
        if not counts: audit["warnings"].append(f"No classification images found in {split}.")
    manifests = _manifest_labels(root); seg = {}; all_hashes = defaultdict(list); basenames = defaultdict(list); pair_classes=Counter()
    # Include the classifier subset in exact duplicate and cross-task basename checks.
    for split in ("train", "test"):
        for f in _files(root / "classification_task" / split):
            if f.suffix.lower() not in IMAGE_EXTS:
                unexpected.append(str(f.relative_to(root))); continue
            basenames[f.stem.lower()].append(f"classification_task/{split}/{f.name}")
            try:
                with Image.open(f) as im: im.verify()
                with Image.open(f) as im: size_counts[f"{im.width}x{im.height}"]+=1
            except (OSError,UnidentifiedImageError,ValueError) as e:
                record={"path":str(f.relative_to(root)),"error":str(e)};malformed.append(record);classification_errors.append(record)
            try: all_hashes[hashlib.sha256(f.read_bytes()).hexdigest()].append(str(f.relative_to(root)))
            except OSError: pass
    for split in ("train", "test"):
        base=root / "segmentation_task" / split
        images=_files(base / "images"); masks=_files(base / "masks")
        img_map={x.stem.lower():x for x in images if x.suffix.lower() in IMAGE_EXTS}
        mask_map={x.stem.lower():x for x in masks if x.suffix.lower() in IMAGE_EXTS}
        matched=sorted(set(img_map)&set(mask_map)); unmatched_images=sorted(set(img_map)-set(mask_map)); unmatched_masks=sorted(set(mask_map)-set(img_map))
        dup=sorted(k for k,v in Counter(x.stem.lower() for x in images+masks).items() if v>2)
        for f in images + masks:
            if f.suffix.lower() not in IMAGE_EXTS: unexpected.append(str(f.relative_to(root)))
            if f.suffix.lower() not in IMAGE_EXTS: continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    size_counts[f"{im.width}x{im.height}"]+=1
                    if "masks" in f.parts:
                        hist=im.convert("L").histogram(); mask_values.update({str(k):n for k,n in enumerate(hist) if n})
                        if sum(hist[1:])==0: empty.append(str(f.relative_to(root)))
            except (OSError, UnidentifiedImageError, ValueError) as e: malformed.append({"path":str(f.relative_to(root)),"error":str(e)})
            try: all_hashes[hashlib.sha256(f.read_bytes()).hexdigest()].append(str(f.relative_to(root)))
            except OSError: pass
            basenames[f.stem.lower()].append(str(f.relative_to(root)))
        for stem in matched:
            img,mask=img_map[stem],mask_map[stem]
            try:
                with Image.open(img) as a, Image.open(mask) as b:
                    if a.size != b.size: malformed.append({"path":str(img.relative_to(root)),"error":f"image-mask dimensions differ: {a.size} vs {b.size}"})
            except Exception: pass
            label=manifests.get(img.name) or manifests.get(mask.name) or "unknown"
            if label=="unknown":
                # BRISC names may encode class; only use explicit known tokens, never arbitrary substring mapping.
                lower=img.stem.lower()
                label=next((x for x in ("glioma","meningioma","pituitary") if lower.startswith(x) or f"_{x}_" in f"_{lower}_"),"unknown")
            pair_classes[label]+=1
        seg[split]={"images":len(images),"masks":len(masks),"matched_pairs":len(matched),"missing_masks":unmatched_images,"missing_images":unmatched_masks,"duplicate_basenames":dup}
    cross_dup=[v for v in all_hashes.values() if len(v)>1]
    def split_tag(path):
        parts=Path(path).parts
        return next((p for p in parts if p in {"train","test","validation"}),None)
    cross_split_dup=[v for v in cross_dup if len({split_tag(p) for p in v if split_tag(p)})>1]
    overlap=sorted(k for k,v in basenames.items() if any("classification_task" in x for x in v) and any("segmentation_task" in x for x in v))
    audit["classification"]["integrity_errors"]=classification_errors
    audit["segmentation"]=seg; audit["paired_by_class"]=dict(pair_classes); audit["non_tumorous_masks_present"]=any(k in ("non_tumorous","no_tumor") and v for k,v in pair_classes.items()); audit["image_dimensions"]=dict(size_counts); audit["mask_unique_values"]=dict(mask_values); audit["empty_masks"]=empty; audit["corrupt_or_dimension_errors"]=malformed; audit["unexpected_files"]=unexpected; audit["sha256_duplicate_groups"]=cross_dup; audit["cross_split_exact_duplicate_groups"]=cross_split_dup; audit["classification_segmentation_basename_overlap_count"]=len(overlap); audit["classification_segmentation_basename_overlap_sample"]=overlap[:20]
    if not audit["segmentation"]["train"]["matched_pairs"]: audit["warnings"].append("No train image/mask pairs found; training is unsafe.")
    if not audit["non_tumorous_masks_present"]: audit["warnings"].append("No non-tumorous/negative segmentation masks detected; segmentation is positive-only and must be classifier-gated.")
    (report_dir/"dataset_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    lines=["# Dataset audit",f"Root: `{root}`","", "## Classification"]
    for split, counts in audit["classification"].items(): lines.append(f"- {split}: {counts}")
    lines += ["", "## Segmentation"]
    for split, values in seg.items(): lines.append(f"- {split}: {values['images']} images, {values['masks']} masks, {values['matched_pairs']} matched pairs")
    lines += [f"- Paired class labels: {dict(pair_classes)}",f"- Negative masks present: {audit['non_tumorous_masks_present']}", f"- Corrupt/dimension errors: {len(malformed)}", f"- SHA-256 duplicate groups: {len(cross_dup)}", "", "## Warnings", *[f"- {w}" for w in audit["warnings"]]]
    (report_dir/"dataset_audit.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return audit
