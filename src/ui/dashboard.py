"""Unified Streamlit interface for the NeuroScan multitask research model."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import CLASSES, MEAN, ROOT, STD
from src.inference.multitask import classify_probabilities, load_multitask_checkpoint, preprocess_image
from src.models.multitask import parameter_count


_checkpoint_setting = Path(os.environ.get("NEUROSCAN_CHECKPOINT", "artifacts/checkpoints/best_multitask_model.pth")).expanduser()
CHECKPOINT_PATH = _checkpoint_setting if _checkpoint_setting.is_absolute() else ROOT / _checkpoint_setting
TEST_METRICS_PATH = ROOT / "artifacts/reports/test_metrics.json"
TEST_REPORT_PATH = ROOT / "artifacts/reports/test_classification_report.json"
DATASET_SUMMARY_PATH = ROOT / "artifacts/reports/dataset_summary.json"
MODEL_COMPARISON_PATH = ROOT / "artifacts/reports/model_comparison.csv"
PLOT_DIR = ROOT / "artifacts/plots"


@st.cache_data(show_spinner=False, max_entries=4)
def _load_json(path_string: str, modified: float):
    del modified
    path = Path(path_string)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False, max_entries=4)
def _load_csv(path_string: str, modified: float):
    del modified
    path = Path(path_string)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


@st.cache_resource(show_spinner="Loading the selected model…")
def _load_model(path_string: str, modified_ns: int, device_name: str):
    del modified_ns
    return load_multitask_checkpoint(path_string, device_name)


def _json_at(path: Path):
    if not path.is_file():
        return None
    return _load_json(str(path), path.stat().st_mtime)


def _csv_at(path: Path):
    if not path.is_file():
        return None
    return _load_csv(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=2)
def _file_sha256(path_string: str, modified_ns: int, size: int):
    del modified_ns, size
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_checkpoint_hash():
    if not CHECKPOINT_PATH.is_file():
        return None
    stat = CHECKPOINT_PATH.stat()
    return _file_sha256(str(CHECKPOINT_PATH), stat.st_mtime_ns, stat.st_size)


def _format_metric(value, percent=False):
    if value is None:
        return "—"
    return f"{float(value):.1%}" if percent else f"{float(value):.3f}"


def _model():
    if not CHECKPOINT_PATH.is_file():
        return None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, metadata = _load_model(
            str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime_ns, device
        )
        return model, metadata, None
    except Exception as exc:  # Show a useful UI error instead of terminating the dashboard.
        return None, None, str(exc)


def _render_overview(model, metadata, model_error):
    st.subheader("Research model at a glance")
    metrics = _json_at(TEST_METRICS_PATH)
    test_summary = _json_at(ROOT / "artifacts/reports/test_summary.json")
    dataset_summary = _json_at(DATASET_SUMMARY_PATH)
    comparison = _csv_at(MODEL_COMPARISON_PATH)
    if metrics and metrics.get("checkpoint_sha256") and CHECKPOINT_PATH.is_file():
        if _current_checkpoint_hash() != metrics["checkpoint_sha256"]:
            st.warning("The installed checkpoint differs from the weights used to create the saved test report.")

    if model_error:
        st.error(f"The selected checkpoint could not be loaded: {model_error}", icon=":material/error:")
    elif metadata is None:
        st.info(
            "Model checkpoint is not installed. The dashboard and saved reports remain available; "
            "add `artifacts/checkpoints/best_multitask_model.pth` to enable image inference. "
            "Model weights are excluded from Git because of their size."
        )
    else:
        st.success("Selected checkpoint loaded and ready for inference.", icon=":material/check_circle:")

    if test_summary and test_summary.get("classification_test") and test_summary.get("segmentation_test"):
        classification = test_summary["classification_test"]
        segmentation = test_summary["segmentation_test"]
        with st.container(horizontal=True):
            st.metric("Test classification accuracy", _format_metric(classification.get("accuracy"), True), border=True)
            st.metric("Test macro F1", _format_metric(classification.get("macro_f1")), border=True)
            st.metric("Test glioma recall", _format_metric(classification.get("glioma_recall_at_threshold"), True), border=True)
            st.metric("Test mean Dice", _format_metric(segmentation.get("dice_mean")), border=True)
            st.metric("Test mean IoU", _format_metric(segmentation.get("iou_mean")), border=True)
        st.caption(
            f"Official held-out test set · {test_summary.get('classification_test_images', '—'):,} classification "
            f"images · {test_summary.get('segmentation_test_pairs', '—'):,} annotated segmentation pairs. "
            "Thresholds were selected on validation data."
        )
    elif metrics is None:
        st.info("No saved official test report is available. Validation metrics are not presented as test results.")
    else:
        st.info("Saved test results are incomplete; run the documented multi-task test evaluation to refresh them.")

    dataset_columns = st.columns(4)
    if test_summary:
        model_name = str(test_summary.get("selected_architecture", "—")).replace("_", " ").title()
        dataset_columns[0].metric("Selected model", model_name)
        dataset_columns[1].metric("Parameters", f"{test_summary.get('number_of_parameters', 0):,}" if test_summary.get("number_of_parameters") else "—")
        dataset_columns[2].metric("Input size", f"{test_summary.get('input_size', '—')} × {test_summary.get('input_size', '—')}")
        dataset_columns[3].metric("Model training time", _training_duration(comparison, model_name))
    else:
        model_name = metadata.get("architecture", "—").replace("_", " ").title() if metadata else "—"
        dataset_columns[0].metric("Selected model", model_name)
        dataset_columns[1].metric("Parameters", f"{parameter_count(model):,}" if model else "—")
        dataset_columns[2].metric("Input size", f"{metadata.get('input_size', '—')} × {metadata.get('input_size', '—')}" if metadata else "—")
        dataset_columns[3].metric("Model training time", _training_duration(comparison, model_name))

    if test_summary:
        st.caption(
            f"Classification threshold: glioma probability ≥ {test_summary.get('glioma_threshold', 'not calibrated')} · "
            f"segmentation threshold: {test_summary.get('segmentation_threshold', 'not calibrated')} · "
            f"inference device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
        )
    else:
        st.caption(
            f"Classification threshold: {metadata.get('glioma_threshold', 'not available') if metadata else 'not available'} · "
            f"segmentation threshold: {metadata.get('mask_threshold', 'not available') if metadata else 'not available'} · "
            f"inference device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
        )

    if dataset := dataset_summary:
        st.subheader("Dataset and split")
        st.write(dataset.get("description", "BRISC2025 MRI classification and segmentation data."))
        cards = st.columns(3)
        cards[0].metric("Official classification training images", f"{dataset.get('classification', {}).get('official_train', '—'):,}")
        cards[1].metric("Official classification test images", f"{dataset.get('classification', {}).get('official_test', '—'):,}")
        cards[2].metric("Official segmentation test pairs", f"{dataset.get('segmentation', {}).get('official_test_pairs', '—'):,}")
        st.caption(dataset.get("split_description", ""))
    elif test_summary:
        st.caption("Dataset audit summary is not included with this clone.")

    if comparison is not None and not comparison.empty:
        st.subheader("Candidate comparison · validation selection")
        display = comparison.copy()
        if "training_seconds" in display:
            display["training_minutes"] = (display.pop("training_seconds") / 60).round(1)
        st.dataframe(display, hide_index=True, alt="Candidate architecture validation comparison")

    st.warning(
        "Research and education only. NeuroScan analyzes a single 2D T1CE slice, is not a medical device, "
        "and cannot diagnose a patient or replace qualified clinical review.",
        icon=":material/health_and_safety:",
    )


def _training_duration(comparison, model_name):
    if comparison is None or "architecture" not in comparison or "training_seconds" not in comparison:
        return "Not recorded"
    architecture = model_name.lower().replace("-", "_").replace(" ", "_")
    selected = comparison.loc[comparison["architecture"].astype(str).str.lower() == architecture]
    if selected.empty:
        return "Not recorded"
    return f"{float(selected.iloc[0]['training_seconds']) / 60:.1f} min"


def _probability_chart(probabilities):
    frame = pd.DataFrame({"Class": CLASSES, "Probability": probabilities})
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4, color="#2B7A78")
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Class:N", sort="-x"),
            tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=180)
    )
    st.altair_chart(chart, width="stretch", theme=None, key="class-probabilities")


def _infer(model, metadata, image):
    size = int(metadata.get("input_size", 224))
    normalization = metadata.get("normalization", {})
    input_tensor = preprocess_image(image, size, normalization.get("mean", MEAN), normalization.get("std", STD))
    device = next(model.parameters()).device
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(output["classification"], dim=1)[0].cpu().numpy()
        mask_probability = torch.sigmoid(output["segmentation"])[0, 0].cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    predicted_index = classify_probabilities(probabilities, metadata.get("glioma_threshold"))
    threshold = float(metadata.get("mask_threshold", 0.5))
    binary = (mask_probability >= threshold).astype(np.uint8)
    return probabilities, predicted_index, binary, elapsed


def _make_overlay(original, binary, opacity):
    mask = Image.fromarray((binary * 255).astype(np.uint8)).resize(original.size, Image.Resampling.NEAREST)
    overlay = original.convert("RGBA")
    tint = Image.new("RGBA", original.size, (235, 45, 65, 0))
    tint.putalpha(mask.point(lambda value: int(value * opacity)))
    return mask, Image.alpha_composite(overlay, tint).convert("RGB")


def _render_single(model, metadata, model_error):
    st.subheader("Analyze one MRI slice")
    st.caption("Use a 2D T1CE slice. The uploaded image is resized and normalized exactly as in the training notebook.")
    upload = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"], key="single-mri")
    opacity = st.slider("Overlay opacity", min_value=0.05, max_value=0.8, value=0.35, step=0.05)
    if not upload:
        st.info("Upload a JPG or PNG slice to begin.")
        return
    data = upload.getvalue()
    image_key = hashlib.sha256(data).hexdigest()
    if len(data) > 20 * 1024 * 1024:
        st.error("Image exceeds the 20 MB upload limit.")
        return
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        if max(original.size) > 8192 or original.width * original.height > 32_000_000:
            raise ValueError("Image dimensions exceed the supported limit.")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not read this image: {exc}")
        return

    if st.session_state.get("prediction_key") != image_key:
        st.session_state.pop("prediction_result", None)
        st.session_state["prediction_key"] = image_key
    left, right = st.columns(2)
    with left:
        st.image(original, caption=f"Original slice · {original.width} × {original.height}", alt="Uploaded MRI slice", width="stretch")
    if model is None:
        message = (
            f"Checkpoint load failed: {model_error}" if model_error else
            "Inference is unavailable because `artifacts/checkpoints/best_multitask_model.pth` is missing. "
            "Run the notebook training/export cells or supply the checkpoint locally."
        )
        st.info(message)
        return

    if st.button("Run analysis", type="primary", icon=":material/radar:"):
        try:
            with st.spinner("Analyzing slice…"):
                probabilities, predicted_index, binary, elapsed = _infer(model, metadata, original)
                st.session_state["prediction_result"] = {
                    "probabilities": probabilities.tolist(),
                    "predicted_index": predicted_index,
                    "binary": binary,
                    "elapsed": elapsed,
                    "image_size": original.size,
                }
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    result = st.session_state.get("prediction_result")
    if result is None or st.session_state.get("prediction_key") != image_key:
        return
    probabilities = np.asarray(result["probabilities"])
    label = CLASSES[result["predicted_index"]]
    mask, overlay = _make_overlay(original, result["binary"], opacity)
    with right:
        st.image(overlay, caption="Predicted mask over MRI", alt="MRI with predicted mask overlay", width="stretch")
    st.subheader("Prediction")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Predicted class", label.replace("_", " ").title())
    metric_cols[1].metric("Class confidence", f"{probabilities[result['predicted_index']]:.1%}")
    metric_cols[2].metric("Inference time", f"{result['elapsed'] * 1000:.0f} ms")
    if label == "non_tumorous":
        st.info("The model assigned the non-tumorous class. Segmentation was still computed; its training data contained tumour-positive masks only.")
    prob_col, mask_col = st.columns([1, 1])
    with prob_col:
        st.markdown("**Class probabilities**")
        _probability_chart(probabilities)
    with mask_col:
        st.markdown("**Binary segmentation mask**")
        st.image(mask, caption=f"Threshold {metadata.get('mask_threshold', 0.5):.2f}", alt="Predicted binary tumour mask", width="stretch")
    mask_buffer = io.BytesIO()
    mask.save(mask_buffer, format="PNG")
    overlay_buffer = io.BytesIO()
    overlay.save(overlay_buffer, format="PNG")
    download_cols = st.columns(2)
    download_cols[0].download_button("Download mask PNG", mask_buffer.getvalue(), "neuroscan_mask.png", "image/png", icon=":material/download:")
    download_cols[1].download_button("Download overlay PNG", overlay_buffer.getvalue(), "neuroscan_overlay.png", "image/png", icon=":material/download:")


def _render_batch(model, metadata, model_error):
    st.subheader("Batch MRI analysis")
    st.caption("Analyze up to 20 JPG or PNG images in one session. Each scan is resized using the training preprocessing.")
    uploads = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch-mri")
    if not uploads:
        st.info("Choose one or more images to prepare a batch.")
        return
    if len(uploads) > 20:
        st.warning("Only the first 20 uploaded images will be analyzed.")
        uploads = uploads[:20]
    if model is None:
        st.info(f"Batch inference is unavailable: {model_error or 'the multi-task checkpoint is missing.'}")
        return
    if not st.button("Analyze batch", type="primary"):
        return
    rows, archive = [], io.BytesIO()
    with st.spinner("Analyzing uploaded images…"):
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for index, upload in enumerate(uploads, start=1):
                try:
                    raw = upload.getvalue()
                    if len(raw) > 20 * 1024 * 1024:
                        raise ValueError("image exceeds the 20 MB limit")
                    with Image.open(io.BytesIO(raw)) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    if max(image.size) > 8192 or image.width * image.height > 32_000_000:
                        raise ValueError("image dimensions exceed the supported limit")
                    probabilities, predicted_index, binary, elapsed = _infer(model, metadata, image)
                    mask, overlay = _make_overlay(image, binary, 0.35)
                    rows.append({
                        "filename": upload.name,
                        "predicted_class": CLASSES[predicted_index],
                        "confidence": float(probabilities[predicted_index]),
                        "inference_ms": round(elapsed * 1000, 1),
                    })
                    output = io.BytesIO()
                    overlay.save(output, format="PNG")
                    zipped.writestr(f"{index:02d}_{Path(upload.name).stem}_overlay.png", output.getvalue())
                    output = io.BytesIO()
                    mask.save(output, format="PNG")
                    zipped.writestr(f"{index:02d}_{Path(upload.name).stem}_mask.png", output.getvalue())
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    rows.append({"filename": upload.name, "error": str(exc)})
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, alt="Batch classification results")
    st.download_button("Download CSV summary", frame.to_csv(index=False), "neuroscan_batch.csv", "text/csv")
    st.download_button("Download masks and overlays", archive.getvalue(), "neuroscan_batch_results.zip", "application/zip")


def _confusion_chart(matrix):
    frame = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).rename_axis("Actual").reset_index().melt(
        id_vars="Actual", var_name="Predicted", value_name="Images"
    )
    heatmap = (
        alt.Chart(frame)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Predicted:N", sort=list(CLASSES)),
            y=alt.Y("Actual:N", sort=list(CLASSES)),
            color=alt.Color("Images:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=["Actual:N", "Predicted:N", "Images:Q"],
        )
    )
    labels = heatmap.mark_text().encode(text="Images:Q", color=alt.value("#15252a"))
    st.altair_chart(heatmap + labels, width="stretch", theme=None, key="test-confusion-matrix")


def _render_evaluation():
    st.subheader("Held-out test evaluation")
    metrics = _json_at(TEST_METRICS_PATH)
    if not metrics or metrics.get("classification", {}).get("split") != "official_test":
        st.info(
            "No verified official test metrics are available in this clone. Validation scores are intentionally "
            "not shown in their place. Add the saved report or run the evaluation command below."
        )
        st.code("python scripts/evaluate_multitask.py --data-root data/raw/brisc2025")
        return

    if metrics.get("checkpoint_sha256") and CHECKPOINT_PATH.is_file():
        if _current_checkpoint_hash() != metrics["checkpoint_sha256"]:
            st.warning("The displayed report was generated with a different checkpoint than the one currently installed.")

    classification = metrics["classification"]
    segmentation = metrics["segmentation"]
    st.caption(
        f"Official test split · thresholds frozen from validation · "
        f"{classification.get('n', '—'):,} classification images and {segmentation.get('n_images', '—'):,} "
        "annotated segmentation images. Segmentation annotations are positive-only."
    )
    with st.container(horizontal=True):
        st.metric("Accuracy", _format_metric(classification.get("accuracy"), True), border=True)
        st.metric("Balanced accuracy", _format_metric(classification.get("balanced_accuracy"), True), border=True)
        st.metric("Macro F1", _format_metric(classification.get("macro_f1")), border=True)
        st.metric("Macro OvR ROC-AUC", _format_metric(classification.get("roc_auc_macro_ovr")), border=True)
        st.metric("Glioma recall at threshold", _format_metric(classification.get("glioma_recall_at_threshold"), True), border=True)

    confusion = classification.get("confusion_matrix")
    if confusion:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("**Confusion matrix · hover for counts**")
            _confusion_chart(confusion)
        with right:
            st.markdown("**Classification report**")
            report = _json_at(TEST_REPORT_PATH)
            if report:
                table = pd.DataFrame(report).T
                table = table.loc[[name for name in CLASSES if name in table.index], ["precision", "recall", "f1-score", "support"]]
                st.dataframe(table, alt="Per-class test precision, recall, F1, and support")
            else:
                st.dataframe(pd.DataFrame(classification.get("per_class", {})).T, alt="Per-class test metrics")

    roc_plot = PLOT_DIR / "test_roc_curves.png"
    legacy_roc = PLOT_DIR / "test_argmax_roc.png"
    chosen_roc = roc_plot if roc_plot.is_file() else legacy_roc
    if chosen_roc.is_file():
        st.image(str(chosen_roc), caption="One-vs-rest ROC curves · official test", alt="Official test ROC curves", width="stretch")
    elif classification.get("roc_auc_macro_ovr") is not None:
        st.caption("ROC-AUC is available in the saved report, but no ROC image is included.")

    st.markdown("**Segmentation quality · annotated positive images only**")
    with st.container(horizontal=True):
        st.metric("Mean Dice", _format_metric(segmentation.get("dice_mean")), border=True)
        st.metric("Median Dice", _format_metric(segmentation.get("dice_median")), border=True)
        st.metric("Mean IoU", _format_metric(segmentation.get("iou_mean")), border=True)
        st.metric("Median IoU", _format_metric(segmentation.get("iou_median")), border=True)
        st.metric("Mask threshold", _format_metric(segmentation.get("threshold")), border=True)

    score_values = segmentation.get("dice_per_image")
    iou_values = segmentation.get("iou_per_image")
    if score_values and iou_values:
        scores = pd.DataFrame({"Dice": score_values, "IoU": iou_values}).reset_index(names="Sample")
        hist = alt.Chart(scores.melt(id_vars="Sample", var_name="Metric", value_name="Score")).mark_bar(opacity=0.8).encode(
            x=alt.X("Score:Q", bin=alt.Bin(maxbins=20), title="Per-image score"),
            y=alt.Y("count():Q", title="Images"),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=["Dice", "IoU"], range=["#2B7A78", "#E0A458"])),
            tooltip=["Metric:N", alt.Tooltip("count():Q", title="Images")],
        ).properties(height=260)
        st.altair_chart(hist, width="stretch", theme=None, key="test-segmentation-score-distribution")

    examples = PLOT_DIR / "test_segmentation_examples.png"
    if examples.is_file():
        st.image(str(examples), caption="Selected official test examples: MRI, reference mask, prediction, and overlay", alt="MRI segmentation test examples", width="stretch")
    comparison = _csv_at(MODEL_COMPARISON_PATH)
    if comparison is not None and not comparison.empty:
        st.subheader("Candidate model comparison · validation only")
        st.dataframe(comparison, hide_index=True, alt="ConvNeXt-Tiny and Swin-Tiny validation comparison")


def _render_methodology(metadata):
    st.subheader("Dataset and method")
    st.markdown(
        "BRISC2025 provides single-slice contrast-enhanced T1-weighted MRI images for four-way classification "
        "and paired tumour masks for binary segmentation. Classification labels are **glioma**, **meningioma**, "
        "**pituitary**, and **non-tumorous**."
    )
    if dataset := _json_at(DATASET_SUMMARY_PATH):
        st.markdown("**Audited dataset counts**")
        st.json(dataset)
    st.markdown(
        "### Training and evaluation\n"
        "- Two shared-encoder candidates are trained: ConvNeXt-Tiny and Swin-Tiny. Each has a four-class head "
        "and a binary mask decoder. The notebook selects a candidate by a joint validation score.\n"
        "- Training and validation records come from the official training partition. Exact-content duplicate "
        "groups are checked across generated splits and against official test records.\n"
        "- Images are converted to RGB, resized to the checkpoint input resolution with bilinear interpolation, "
        "and normalized with ImageNet channel means and standard deviations. Masks use nearest-neighbor resizing "
        "and are binarized above grayscale 127.\n"
        "- The glioma operating threshold and segmentation-mask threshold are calibrated using validation data. "
        "The official test set is reserved for final evaluation.\n"
        "- Segmentation test scores cover only paired annotated images. The dataset audit found no empty/negative "
        "masks, so performance on normal slices is not established."
    )
    st.markdown(
        "### Limitations\n"
        "The model sees one 2D slice at a time and does not analyze 3D volumes or anatomical regions. Source "
        "patient identifiers are not available to guarantee patient-level independence. Dataset labels are not "
        "clinical conclusions, and the model is not validated for diagnosis or treatment."
    )
    st.markdown("### Model package")
    checkpoint_info = {
        "status": "available" if CHECKPOINT_PATH.is_file() else "not installed",
        "relative_path": "artifacts/checkpoints/best_multitask_model.pth",
        "size_mb": round(CHECKPOINT_PATH.stat().st_size / (1024 * 1024), 1) if CHECKPOINT_PATH.is_file() else None,
        "sha256": _current_checkpoint_hash(),
    }
    st.json(checkpoint_info)
    if metadata:
        visible = {
            "architecture": metadata.get("architecture"),
            "input_size": metadata.get("input_size"),
            "glioma_threshold": metadata.get("glioma_threshold"),
            "segmentation_threshold": metadata.get("mask_threshold"),
            "normalization": metadata.get("normalization"),
            "torch_version": metadata.get("torch"),
            "torchvision_version": metadata.get("torchvision"),
        }
        st.json(visible)
    else:
        st.info("Checkpoint metadata is unavailable because no model weights are installed.")


def render_dashboard():
    st.title("NeuroScan")
    st.caption("Brain MRI classification and tumour-mask research workspace")
    st.warning(
        "Under active development · For research and education only; not a medical device or clinical diagnosis.",
        icon=":material/science:",
    )

    model, metadata, model_error = _model()
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    with st.sidebar:
        st.header("NeuroScan")
        st.caption("Shared encoder · classification + segmentation")
        section = st.segmented_control(
            "Dashboard section",
            options=["Overview", "MRI workspace", "Evaluation", "Method & model"],
            default="Overview",
            key="dashboard-section",
            width="stretch",
        )
        st.divider()
        st.markdown("**Runtime**")
        st.write(f"Inference device: `{device}`")
        if metadata:
            st.write("Checkpoint: `ready`")
        elif model_error:
            st.write("Checkpoint: `load failed`")
        else:
            st.write("Checkpoint: `not installed`")
        if metadata:
            st.caption(f"Architecture: {metadata.get('architecture', 'unknown').replace('_', ' ').title()}")
        elif model_error:
            st.caption(model_error)

    section = section or "Overview"
    if section == "Overview":
        _render_overview(model, metadata, model_error)
    elif section == "MRI workspace":
        mode = st.segmented_control("Analysis mode", options=["Single scan", "Batch scans"], default="Single scan", key="analysis-mode")
        if mode == "Batch scans":
            _render_batch(model, metadata, model_error)
        else:
            _render_single(model, metadata, model_error)
    elif section == "Evaluation":
        _render_evaluation()
    else:
        _render_methodology(metadata)
