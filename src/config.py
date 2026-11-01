from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("glioma", "meningioma", "pituitary", "non_tumorous")
FOLDER_LABELS = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary", "no_tumor": "non_tumorous", "non_tumorous": "non_tumorous"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

