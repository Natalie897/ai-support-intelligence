"""BANKING77 data sources, CSV validation, and verified downloader."""
from pathlib import Path
import hashlib
import pandas as pd
import urllib.request

SOURCE_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
SOURCE_URL = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{SOURCE_COMMIT}/banking_data"
EXPECTED_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def download_banking77(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename, digest in EXPECTED_SHA256.items():
        target = destination / filename
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        with urllib.request.urlopen(f"{SOURCE_URL}/{filename}", timeout=45) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch for {filename}; refusing to use downloaded data")
        target.write_bytes(payload)


def load_split(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"text", "category"}.issubset(frame.columns):
        raise ValueError(f"Expected text and category columns: {path}")
    frame = frame[["text", "category"]].copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip()
    if (frame["text"] == "").any() or (frame["category"] == "").any():
        raise ValueError(f"Empty text/category in {path}")
    if frame.empty:
        raise ValueError(f"Dataset empty: {path}")
    return frame
