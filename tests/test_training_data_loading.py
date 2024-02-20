import pytest

from ragatouille import RAGTrainer


@pytest.fixture
def rag_trainer():
    # Setup for a RAGTrainer instance
    instance = RAGTrainer(
        model_name="test_model", pretrained_model_name="bert-base-uncased"
    )
    return instance


@pytest.mark.parametrize(
    "input_data,pairs_with_labels,expected_queries,expected_collection",
    [
        # Unlabeled pairs
        (
            [("Query1", "Document1"), ("Query2", "Document2")],
            False,
            {"Query1", "Query2"},
            ["Document1", "Document2"],
        ),
        # Labeled pairs
        (
            [
                ("Query1", "Document1Pos", 1),
                ("Query1", "Document1Neg", 0),
                ("Query2", "Document2", 0),
            ],
            True,
            {"Query1", "Query2"},
            ["Document1Pos", "Document1Neg", "Document2"],
        ),
        # Triplets
        (
            [("Query1", "Positive Doc", "Negative Doc")],
            False,
            {"Query1"},
            ["Positive Doc", "Negative Doc"],
        ),
    ],
)
def test_prepare_training_data(
    rag_trainer, input_data, pairs_with_labels, expected_queries, expected_collection
):
    rag_trainer.prepare_training_data(
        raw_data=input_data, pairs_with_labels=pairs_with_labels
    )

    assert rag_trainer.queries == expected_queries

    assert len(rag_trainer.collection) == len(expected_collection)
    assert set(rag_trainer.collection) == set(expected_collection)


def test_prepare_training_data_with_all_documents(rag_trainer):
    input_data = [("Query1", "Document1")]
    all_documents = ["Document2", "Document3"]

    rag_trainer.prepare_training_data(raw_data=input_data, all_documents=all_documents)

    assert rag_trainer.queries == {"Query1"}

    assert len(rag_trainer.collection) == 3
    assert set(rag_trainer.collection) == {"Document1", "Document2", "Document3"}


def test_prepare_training_data_invalid_input(rag_trainer):
    # Providing an invalid input format
    with pytest.raises(ValueError):
        rag_trainer.prepare_training_data(raw_data=[("Query1")])  # Missing document
