import pandas as pd
import pytest

from ml.data import load_split


def test_load_split_accepts_csv(tmp_path):
    target = tmp_path / "examples.csv"
    target.write_text('text,category\n" My card was lost ",lost_or_stolen_card\n', encoding="utf-8")
    frame = load_split(target)
    assert frame.iloc[0].text == "My card was lost"


def test_load_split_rejects_empty_text(tmp_path):
    target = tmp_path / "bad.csv"
    pd.DataFrame({"text": [""], "category": ["cards"]}).to_csv(target, index=False)
    with pytest.raises(ValueError, match="Empty"):
        load_split(target)


def test_load_split_requires_schema(tmp_path):
    target = tmp_path / "bad.csv"
    target.write_text("request,label\nHi,greeting\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected"):
        load_split(target)
