from unittest.mock import MagicMock

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


def test_export_training_data_writes_utf8(tmp_path, collection, queries):
    """Non-ASCII text must survive export on Windows (cp1252) locales (#268)."""
    processor = TrainingDataProcessor(collection, queries, None)
    processor.query_map = {"Merhaba ıüöğç": 0}
    processor.passage_map = {"İstanbul ğ": 0}
    processor.training_triplets = []

    processor.export_training_data(tmp_path)

    queries_tsv = (tmp_path / "queries.train.colbert.tsv").read_text(encoding="utf-8")
    corpus_tsv = (tmp_path / "corpus.train.colbert.tsv").read_text(encoding="utf-8")
    assert "Merhaba ıüöğç" in queries_tsv
    assert "İstanbul ğ" in corpus_tsv

