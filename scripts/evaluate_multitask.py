"""Reproduce held-out test metrics for the selected multi-task checkpoint."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "neuroscan-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader, Dataset

from src.config import CLASSES, FOLDER_LABELS, MEAN, STD
from src.data.audit import discover_root
from src.data.dataset import classification_records, segmentation_records
from src.inference.multitask import load_multitask_checkpoint, preprocess_image, classify_probabilities
from src.models.multitask import parameter_count


class OfficialTestDataset(Dataset):
    def __init__(self, rows, task, size):
        self.rows, self.task, self.size = rows, task, int(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.task == "classification":
            with Image.open(row["path"]) as image:
                image = ImageOps.exif_transpose(image)
                tensor = preprocess_image(image, self.size)
            return tensor, CLASSES.index(FOLDER_LABELS.get(row["label"], row["label"]))
        with Image.open(row["image"]) as image:
            tensor = preprocess_image(ImageOps.exif_transpose(image), self.size)
        with Image.open(row["mask"]) as mask:
            target = mask.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
            target = torch.from_numpy((np.asarray(target, dtype=np.uint8).copy() > 127).astype(np.float32))[None]
        return tensor, target


def _classification_metrics(labels, probabilities, threshold, split="official_test"):
    predictions = np.asarray([classify_probabilities(row, threshold) for row in probabilities])
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=range(len(CLASSES)), zero_division=0
    )
    confusion = confusion_matrix(labels, predictions, labels=range(len(CLASSES)))
    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i, name in enumerate(CLASSES)
    }
    report = classification_report(
        labels, predictions, labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    y_one_hot = np.eye(len(CLASSES), dtype=np.uint8)[labels]
    try:
        macro_auc = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = None
    try:
        per_class_auc = {
            name: float(roc_auc_score(y_one_hot[:, i], probabilities[:, i]))
            for i, name in enumerate(CLASSES)
        }
    except ValueError:
        per_class_auc = {name: None for name in CLASSES}
    argmax_predictions = probabilities.argmax(axis=1)
    argmax_precision, argmax_recall, argmax_f1, _ = precision_recall_fscore_support(
        labels, argmax_predictions, labels=range(len(CLASSES)), zero_division=0
    )
    argmax_metrics = {
        "accuracy": float(accuracy_score(labels, argmax_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, argmax_predictions)),
        "macro_f1": float(f1_score(labels, argmax_predictions, average="macro", zero_division=0)),
        "glioma_recall": float(argmax_recall[0]),
        "confusion_matrix": confusion_matrix(labels, argmax_predictions, labels=range(len(CLASSES))).tolist(),
    }
    return predictions, {
        "split": split,
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "roc_auc_macro_ovr": macro_auc,
        "roc_auc_per_class_ovr": per_class_auc,
        "glioma_threshold": float(threshold) if threshold is not None else None,
        "glioma_recall_at_threshold": float(recall[0]),
        "argmax_glioma_recall": float(argmax_recall[0]),
        "threshold_source": "validation" if threshold is not None else "none; argmax",
        "argmax": argmax_metrics,
    }, report


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segmentation_metrics(probabilities, targets, threshold, split="official_test"):
    predicted = probabilities >= float(threshold)
    truth = targets >= 0.5
    axes = tuple(range(1, predicted.ndim))
    intersection = np.logical_and(predicted, truth).sum(axis=axes)
    predicted_count = predicted.sum(axis=axes)
    truth_count = truth.sum(axis=axes)
    union = np.logical_or(predicted, truth).sum(axis=axes)
    dice = np.divide(2 * intersection, predicted_count + truth_count,
                     out=np.ones_like(intersection, dtype=float), where=(predicted_count + truth_count) > 0)
    iou = np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union > 0)
    tp = np.logical_and(predicted, truth).sum()
    fp = np.logical_and(predicted, ~truth).sum()
    fn = np.logical_and(~predicted, truth).sum()
    tn = np.logical_and(~predicted, ~truth).sum()
    return {
        "split": split,
        "n_images": int(len(dice)),
        "dice_per_image": dice.astype(float).tolist(),
        "iou_per_image": iou.astype(float).tolist(),
        "dice_mean": float(dice.mean()),
        "dice_median": float(np.median(dice)),
        "iou_mean": float(iou.mean()),
        "iou_median": float(np.median(iou)),
        "pixel_precision": float(tp / max(1, tp + fp)),
        "pixel_recall": float(tp / max(1, tp + fn)),
        "pixel_specificity": float(tn / max(1, tn + fp)),
        "threshold": float(threshold),
        "threshold_source": "validation",
        "empty_masks_in_test": int(np.count_nonzero(truth_count == 0)),
        "annotation_scope": "Official test segmentation pairs; dataset audit found no empty/negative masks.",
    }


def _save_plots(labels, probabilities, predictions, segmentation, targets, images, threshold, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=CLASSES,
           yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual", title="Official test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(plot_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    one_hot = np.eye(len(CLASSES), dtype=np.uint8)[labels]
    for index, name in enumerate(CLASSES):
        if len(np.unique(one_hot[:, index])) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        ax.plot(false_positive_rate, true_positive_rate,
                label=f"{name} (AUC {auc(false_positive_rate, true_positive_rate):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Official test one-vs-rest ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "test_roc_curves.png", dpi=160)
    plt.close(fig)

    seg_metrics = _segmentation_metrics(segmentation, targets, threshold)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(seg_metrics["dice_per_image"], bins=20, range=(0, 1), color="#2B7A78")
    axes[0].set(title="Test Dice distribution", xlabel="Per-image Dice", ylabel="Images")
    axes[1].hist(seg_metrics["iou_per_image"], bins=20, range=(0, 1), color="#E0A458")
    axes[1].set(title="Test IoU distribution", xlabel="Per-image IoU", ylabel="Images")
    fig.tight_layout()
    fig.savefig(plot_dir / "test_dice_iou_distribution.png", dpi=160)
    plt.close(fig)

    chosen = sorted(set([int(np.argmin(seg_metrics["dice_per_image"])),
                         int(np.argsort(seg_metrics["dice_per_image"])[len(seg_metrics["dice_per_image"]) // 2]),
                         int(np.argmax(seg_metrics["dice_per_image"]))]))
    fig, axes = plt.subplots(len(chosen), 4, figsize=(12, 3.2 * len(chosen)), squeeze=False)
    for row, index in enumerate(chosen):
        source = images[index]
        source = np.moveaxis(source, 0, -1)
        source = (source * np.asarray(STD) + np.asarray(MEAN)).clip(0, 1)
        mask_truth = targets[index, 0] >= 0.5
        mask_pred = segmentation[index, 0] >= threshold
        axes[row, 0].imshow(source); axes[row, 0].set_title("MRI slice")
        axes[row, 1].imshow(mask_truth, cmap="gray"); axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(mask_pred, cmap="gray"); axes[row, 2].set_title("Predicted mask")
        axes[row, 3].imshow(source); axes[row, 3].imshow(mask_pred, cmap="Reds", alpha=0.35)
        axes[row, 3].set_title(f"Overlay · Dice {seg_metrics['dice_per_image'][index]:.3f}")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(plot_dir / "test_segmentation_examples.png", dpi=160)
    plt.close(fig)
    return seg_metrics


def run_evaluation(data_root, checkpoint_path, output_dir, plot_dir, batch_size=8):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    root = discover_root(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_multitask_checkpoint(checkpoint_path, device)
    size = int(metadata.get("input_size", 224))
    threshold = metadata.get("glioma_threshold")
    mask_threshold = float(metadata.get("mask_threshold", 0.5))
    class_rows = classification_records(root, "test")
    seg_rows = segmentation_records(root, "test")
    if not class_rows or not seg_rows:
        raise RuntimeError("Official test classification images and segmentation pairs are both required.")
    class_loader = DataLoader(OfficialTestDataset(class_rows, "classification", size), batch_size=batch_size)
    seg_loader = DataLoader(OfficialTestDataset(seg_rows, "segmentation", size), batch_size=batch_size)

    labels, class_probabilities = [], []
    with torch.inference_mode():
        for images, targets in class_loader:
            logits = model(images.to(device))["classification"]
            class_probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(targets.numpy())
    labels = np.asarray(labels, dtype=np.int64)
    class_probabilities = np.asarray(class_probabilities, dtype=np.float64)
    class_predictions, classification, report = _classification_metrics(labels, class_probabilities, threshold)

    segmentation_probabilities, segmentation_targets, input_images = [], [], []
    with torch.inference_mode():
        for images, targets in seg_loader:
            logits = model(images.to(device))["segmentation"]
            segmentation_probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            segmentation_targets.extend(targets.numpy())
            input_images.extend(images.numpy())
    segmentation_probabilities = np.asarray(segmentation_probabilities, dtype=np.float32)
    segmentation_targets = np.asarray(segmentation_targets, dtype=np.float32)
    segmentation = _save_plots(labels, class_probabilities, class_predictions,
                                segmentation_probabilities, segmentation_targets, input_images,
                                mask_threshold, Path(plot_dir))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps({
        "selected_architecture": metadata["architecture"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "torch_version": str(torch.__version__),
        "torchvision_version": __import__("torchvision").__version__,
        "classification": classification,
        "segmentation": segmentation,
        "glioma_threshold": threshold,
        "mask_threshold": mask_threshold,
        "threshold_source": "validation",
        "device": str(device),
        "parameter_count": parameter_count(model),
        "segmentation_annotation_scope": segmentation["annotation_scope"],
    }, indent=2))
    (output_dir / "test_classification_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "sample_index": np.arange(len(labels)),
        "true_label": [CLASSES[i] for i in labels],
        "predicted_label": [CLASSES[i] for i in class_predictions],
        **{f"probability_{name}": class_probabilities[:, i] for i, name in enumerate(CLASSES)},
    }).to_csv(output_dir / "test_class_probabilities.csv", index=False)
    (output_dir / "test_segmentation_metrics.json").write_text(json.dumps(segmentation, indent=2))
    summary = {
        "selected_architecture": metadata["architecture"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "input_size": size,
        "number_of_parameters": parameter_count(model),
        "glioma_threshold": threshold,
        "segmentation_threshold": mask_threshold,
        "threshold_source": "validation",
        "classification_test_images": len(class_rows),
        "segmentation_test_pairs": len(seg_rows),
        "classification_test": {key: classification[key] for key in
                                 ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_macro_ovr", "glioma_recall_at_threshold", "argmax")},
        "segmentation_test": {key: segmentation[key] for key in
                              ("dice_mean", "dice_median", "iou_mean", "iou_median")},
        "negative_segmentation_masks_present": segmentation["empty_masks_in_test"] > 0,
    }
    (output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("NEUROSCAN_DATA_ROOT", str(ROOT / "data/raw")))
    parser.add_argument("--checkpoint", default=os.environ.get("NEUROSCAN_CHECKPOINT", str(ROOT / "artifacts/checkpoints/best_multitask_model.pth")))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/reports"))
    parser.add_argument("--plot-dir", default=str(ROOT / "artifacts/plots"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.data_root, args.checkpoint, args.output_dir, args.plot_dir, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
