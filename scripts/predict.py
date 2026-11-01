import argparse,json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.pipeline import analyze_image
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("image");p.add_argument("--classifier",default="artifacts/checkpoints/classifier_best.pth");p.add_argument("--segmenter",default="artifacts/checkpoints/segmenter_attention_unet_best.pth");a=p.parse_args();r,_=analyze_image(a.image,a.classifier,a.segmenter);print(json.dumps(r.to_dict(),indent=2))
