from pathlib import Path

def download_dataset(handle="briscdataset/brisc2025", output_dir="data/raw"):
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `pip install -r requirements.txt` (kagglehub is required).") from exc
    try:
        path = kagglehub.dataset_download(handle, output_dir=str(Path(output_dir)))
    except Exception as exc:
        raise RuntimeError("Kaggle download failed. Configure KAGGLE_API_TOKEN or authenticate with kagglehub.login(), then retry.") from exc
    return Path(path)
