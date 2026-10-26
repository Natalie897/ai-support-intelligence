from pathlib import Path
from ml.data import download_banking77

if __name__ == "__main__":
    download_banking77(Path("data/banking77"))
    print("Verified BANKING77 files downloaded to data/banking77")
