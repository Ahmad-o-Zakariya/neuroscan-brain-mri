import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import ROOT
from src.inference.multitask import classify_probabilities, load_multitask_checkpoint, preprocess_image
from src.models.multitask import MultiTaskModel, parameter_count
from scripts.evaluate_multitask import _classification_metrics, _segmentation_metrics


def test_preprocessing_matches_rgb_resize_and_imagenet_normalization():
    image = Image.new("RGB", (8, 4), (128, 64, 255))
    tensor = preprocess_image(image, size=12)
    assert tuple(tensor.shape) == (3, 12, 12)
    expected = (np.asarray([128, 64, 255], dtype=np.float32) / 255 - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    np.testing.assert_allclose(tensor[:, 4, 4].numpy(), expected, rtol=1e-5, atol=1e-5)


def test_saved_glioma_threshold_uses_remaining_classes_as_fallback():
    assert classify_probabilities([.90, .04, .03, .03], .95) == 1
    assert classify_probabilities([.96, .01, .02, .01], .95) == 0
    assert classify_probabilities([.90, .04, .03, .03], None) == 0


@pytest.mark.parametrize("architecture", ["convnext_tiny", "swin_tiny"])
def test_multitask_architectures_return_class_and_mask_outputs(architecture):
    torch.set_num_threads(2)
    model = MultiTaskModel(architecture).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output["classification"].shape == (1, 4)
    assert output["segmentation"].shape == (1, 1, 224, 224)
    assert parameter_count(model) > 20_000_000


def test_actual_selected_checkpoint_loads_safely_when_available():
    checkpoint = ROOT / "artifacts/checkpoints/best_multitask_model.pth"
    if not checkpoint.is_file():
        pytest.skip("The large trained checkpoint is not included in a clean clone.")
    model, metadata = load_multitask_checkpoint(checkpoint)
    assert metadata["architecture"] == "convnext_tiny"
    assert parameter_count(model) == 28_900_101


def test_multitask_evaluation_calculates_threshold_and_per_image_metrics():
    y = np.asarray([0, 1, 2, 3])
    probability = np.asarray([
        [.85, .06, .05, .04],
        [.08, .80, .07, .05],
        [.06, .08, .80, .06],
        [.03, .07, .05, .85],
    ])
    _, classification, report = _classification_metrics(y, probability, .90, split="synthetic_test_fixture")
    assert classification["split"] == "synthetic_test_fixture"
    assert classification["accuracy"] == .75
    assert classification["argmax"]["accuracy"] == 1.0
    assert classification["roc_auc_macro_ovr"] == 1.0
    assert classification["glioma_recall_at_threshold"] == 0.0
    assert report["glioma"]["recall"] == 0.0

    masks = np.ones((2, 1, 4, 4), dtype=bool)
    predicted = masks.astype(np.float32)
    result = _segmentation_metrics(predicted, masks, .5, split="synthetic_test_fixture")
    assert result["dice_mean"] == 1.0
    assert result["iou_median"] == 1.0
    assert result["empty_masks_in_test"] == 0


def test_dashboard_starts_without_checkpoint_from_another_working_directory(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEUROSCAN_CHECKPOINT", str(ROOT / "missing-checkpoint-for-test.pth"))
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert [item.value for item in app.title] == ["NeuroScan"]
    assert any("checkpoint is not installed" in item.value.lower() for item in app.info)
    assert any(item.label == "Test classification accuracy" for item in app.metric)
    app.segmented_control[0].set_value("Evaluation").run()
    assert not app.exception
    assert any(item.value == "Held-out test evaluation" for item in app.subheader)


def test_committed_reports_do_not_contain_this_machine_path():
    for filename in ("dataset_summary.json", "test_summary.json", "test_metrics.json"):
        path = ROOT / "artifacts/reports" / filename
        if not path.is_file():
            pytest.skip("Reports are not generated yet.")
        content = path.read_text()
        assert str(ROOT) not in content
        json.loads(content)
