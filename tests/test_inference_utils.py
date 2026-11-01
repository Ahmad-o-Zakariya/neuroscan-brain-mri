import pytest
np=pytest.importorskip("numpy")
from PIL import Image
from src.inference.pipeline import _components,annotated_image

def test_components_and_overlay_alignment():
 mask=np.zeros((20,30),dtype=bool);mask[3:7,8:12]=1;boxes,centers=_components(mask);assert boxes[0]["x"]==8 and centers[0]["y"]==4.5
 out=annotated_image(Image.new("RGB",(30,20)),mask,boxes=boxes,centroids=centers);assert out.size==(30,20)

def test_missing_classifier_checkpoint():
 from src.inference.pipeline import analyze_image
 with pytest.raises(FileNotFoundError):analyze_image(Image.new("L",(20,20)),"missing.pth")
