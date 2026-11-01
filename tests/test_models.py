import pytest
torch=pytest.importorskip("torch")
from src.models.networks import UNet,build_segmenter

def test_unet_output_shape():
 for architecture in ("unet","attention_unet"):
  model=build_segmenter(architecture,base_channels=2);assert model(torch.randn(1,1,64,64)).shape==(1,1,64,64)
