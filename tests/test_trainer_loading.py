from colbert.infra import ColBERTConfig

from ragatouille import RAGTrainer


def test_finetune():
    """Ensure that the initially loaded config is the one from the pretrained model."""
    trainer = RAGTrainer(
        model_name="test",
        pretrained_model_name="colbert-ir/colbertv2.0",
        language_code="en",
    )
    trainer_config = trainer.model.config

    assert ColBERTConfig() != trainer_config
    assert trainer_config.query_token == "[Q]"
    assert trainer_config.doc_token == "[D]"
    assert trainer_config.nbits == 1
    assert trainer_config.kmeans_niters == 20
    assert trainer_config.lr == 1e-05
    assert trainer_config.relu is False
    assert trainer_config.nway == 64
    assert trainer_config.doc_maxlen == 180
    assert trainer_config.use_ib_negatives is True
    assert trainer_config.name == "kldR2.nway64.ib"


def test_raw_model():
    """Ensure that the default ColBERT configuration is properly loaded when initialising from a BERT-like model"""  # noqa: E501
    trainer = RAGTrainer(
        model_name="test",
        pretrained_model_name="bert-base-uncased",
        language_code="en",
    )
    trainer_config = trainer.model.config

    default_config = ColBERTConfig()

    assert trainer_config.query_token == default_config.query_token
    assert trainer_config.doc_token == default_config.doc_token
    assert trainer_config.nway == default_config.nway
    assert trainer_config.doc_maxlen == default_config.doc_maxlen
    assert trainer_config.bsize == default_config.bsize
    assert trainer_config.use_ib_negatives == default_config.use_ib_negatives
