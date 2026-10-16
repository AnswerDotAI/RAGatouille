from unittest.mock import MagicMock

import builtins

import pytest

from ragatouille.data import TrainingDataProcessor


@pytest.fixture
def collection():
    return ["doc1", "doc2", "doc3"]


@pytest.fixture
def queries():
    return ["query1", "query2"]


def test_process_raw_data_without_miner(collection, queries):
    processor = TrainingDataProcessor(collection, queries, None)
    processor._process_raw_pairs = MagicMock(return_value=None)

    processor.process_raw_data(
        raw_data=[], data_type="pairs", data_dir="./", mine_hard_negatives=False
    )

    processor._process_raw_pairs.assert_called_once()


def test_process_raw_data_with_miner(collection, queries):
    negative_miner = MagicMock()
    processor = TrainingDataProcessor(collection, queries, negative_miner)
    processor._process_raw_pairs = MagicMock(return_value=None)

    processor.process_raw_data(raw_data=[], data_type="pairs", data_dir="./")

    processor._process_raw_pairs.assert_called_once()


def test_export_training_data_writes_non_ascii_when_default_encoding_is_ascii(
    tmp_path, monkeypatch
):
    """Windows uses a locale encoding such as cp1252; Turkish ı is not in that set."""
    real_open = builtins.open

    def ascii_default_open(file, mode="r", *args, **kwargs):
        if "b" not in str(mode) and kwargs.get("encoding") is None:
            kwargs["encoding"] = "ascii"
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(
        "ragatouille.data.training_data_processor.open", ascii_default_open
    )

    processor = TrainingDataProcessor(
        collection=["COVID-19 data with Turkish ı"],
        queries=["GPT-4 and COVID-19"],
        negative_miner=None,
    )
    processor.training_triplets = []
    processor.export_training_data(tmp_path)

    queries = (tmp_path / "queries.train.colbert.tsv").read_text(encoding="utf-8")
    corpus = (tmp_path / "corpus.train.colbert.tsv").read_text(encoding="utf-8")
    assert "GPT-4 and COVID-19" in queries
    assert "COVID-19 data with Turkish ı" in corpus


def test_export_training_data_ascii_content(tmp_path):
    processor = TrainingDataProcessor(
        collection=["doc one"],
        queries=["query one"],
        negative_miner=None,
    )
    processor.training_triplets = []
    processor.export_training_data(tmp_path)

    queries = (tmp_path / "queries.train.colbert.tsv").read_text(encoding="utf-8")
    corpus = (tmp_path / "corpus.train.colbert.tsv").read_text(encoding="utf-8")
    assert "query one" in queries
    assert "doc one" in corpus

