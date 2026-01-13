import os
import requests
from pathlib import Path

def download_if_missing(model_name, url):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / model_name

    if model_path.exists():
        return str(model_path)

    print(f"Downloading {model_name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(model_path)
