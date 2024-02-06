import pytest
from unittest.mock import MagicMock

from ragatouille.data import TrainingDataProcessor


@pytest.fixture
def collection():
    return ['doc1', 'doc2', 'doc3']


@pytest.fixture
def queries():
    return ['query1', 'query2']


def test_process_raw_data_without_miner(collection, queries):
    processor = TrainingDataProcessor(collection, queries, None)
    processor._process_raw_pairs = MagicMock(return_value=None)

    processor.process_raw_data(raw_data=[], data_type="pairs", data_dir="./", mine_hard_negatives=False)

    processor._process_raw_pairs.assert_called_once()


def test_process_raw_data_with_miner(collection, queries):
    negative_miner = MagicMock()
    processor = TrainingDataProcessor(collection, queries, negative_miner)
    processor._process_raw_pairs = MagicMock(return_value=None)

    processor.process_raw_data(raw_data=[], data_type="pairs", data_dir="./")

    processor._process_raw_pairs.assert_called_once()