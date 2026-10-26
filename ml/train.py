"""Compare baseline models, select by validation macro-F1, then evaluate once on test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import sklearn

from ml.data import download_banking77, load_split


def build_pipeline(name: str) -> Pipeline:
    if name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1200, C=4.0, random_state=42)
    elif name == "multinomial_nb":
        classifier = MultinomialNB(alpha=0.5)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", classifier),
    ])


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def train(train_csv: Path, test_csv: Path, artifact_dir: Path, demo: bool = False) -> dict:
    train_frame = load_split(train_csv)
    test_frame = load_split(test_csv)
    unseen_labels = set(test_frame.category) - set(train_frame.category)
    if unseen_labels:
        raise ValueError(f"Test includes labels missing from training: {sorted(unseen_labels)}")
    if len(set(train_frame.category)) < 2:
        raise ValueError("At least 2 categories required")
    # Use a stratified validation set; official test remains untouched for model selection.
    x_train, x_val, y_train, y_val = train_test_split(
        train_frame.text, train_frame.category,
        test_size=0.2 if demo else 0.15,
        random_state=42, stratify=train_frame.category,
    )
    candidates = {}
    for name in ("multinomial_nb", "logistic_regression"):
        candidate = build_pipeline(name)
        candidate.fit(x_train, y_train)
        candidates[name] = metrics(y_val, candidate.predict(x_val))
    winner = max(candidates, key=lambda n: (candidates[n]["macro_f1"], candidates[n]["accuracy"]))
    model = build_pipeline(winner)
    model.fit(train_frame.text, train_frame.category)
    final_metrics = metrics(test_frame.category, model.predict(test_frame.text))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib", compress=3)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "model_version": version,
        "dataset": "demo_synthetic" if demo else "BANKING77",
        "dataset_language": "en",
        "mode": "demo" if demo else "real",
        "algorithm": winner,
        "validation_results": candidates,
        "test_results": final_metrics,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "num_classes": len(model.classes_),
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "notes": "Scores are uncalibrated model probabilities, not a safety guarantee.",
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a BANKING77 intent classifier")
    parser.add_argument("--demo", action="store_true", help="Offline smoke training on tiny illustrative data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/banking77"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-download", action="store_true", help="Use existing BANKING77 CSV files")
    args = parser.parse_args()
    if args.demo:
        root = Path("data")
        train_csv, test_csv = root / "demo_train.csv", root / "demo_test.csv"
    else:
        if not args.no_download:
            download_banking77(args.data_dir)
        train_csv, test_csv = args.data_dir / "train.csv", args.data_dir / "test.csv"
    result = train(train_csv, test_csv, args.artifact_dir, demo=args.demo)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
