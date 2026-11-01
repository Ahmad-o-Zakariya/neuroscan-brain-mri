import random, numpy as np, torch
from pathlib import Path
import json

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_report():
    d="cuda" if torch.cuda.is_available() else "cpu"
    return {"device":d,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

def save_checkpoint(path,model,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"metadata":metadata},path)
    path.with_suffix(".json").write_text(json.dumps(metadata,indent=2))

def load_checkpoint(path,model,device="cpu",expected_architecture=None):
    ck=torch.load(path,map_location=device,weights_only=True)
    if expected_architecture and ck.get("metadata",{}).get("architecture")!=expected_architecture: raise ValueError("Checkpoint architecture metadata mismatch")
    model.load_state_dict(ck["state_dict"],strict=True); return ck.get("metadata",{})
