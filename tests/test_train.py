import json

from app.inference import IntentClassifier
from ml.train import build_pipeline, train


def test_training_pipeline_end_to_end(tmp_path):
    output = tmp_path / "models"
    report = train(
        train_csv=__import__("pathlib").Path("data/demo_train.csv"),
        test_csv=__import__("pathlib").Path("data/demo_test.csv"),
        artifact_dir=output,
        demo=True,
    )
    assert report["mode"] == "demo"
    assert report["num_classes"] == 6
    assert 0 <= report["test_results"]["macro_f1"] <= 1
    assert json.loads((output / "metrics.json").read_text())["algorithm"] in {"multinomial_nb", "logistic_regression"}
    classifier = IntentClassifier.load(output / "model.joblib")
    assert len(classifier.predict("My card was stolen", top_k=2)) == 2


def test_invalid_model_name():
    import pytest
    with pytest.raises(ValueError):
        build_pipeline("not-a-model")
