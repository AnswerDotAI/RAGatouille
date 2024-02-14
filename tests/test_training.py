import pathlib
import random
import signal
from contextlib import contextmanager

import pytest

from ragatouille import RAGTrainer
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter

DATA_DIR = pathlib.Path(__file__).parent / "data"


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Time limit context manager as in https://stackoverflow.com/a/601168"""

    def signal_handler(_, __):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@pytest.mark.slow
def test_training(tmp_path):
    """This test is based on the content of examples/02-basic_training.ipynb
    and mainly tests that there are no exceptions which can happen e.g. due
    to bugs in data processing.
    """
    trainer = RAGTrainer(
        model_name="GhibliColBERT",
        pretrained_model_name="colbert-ir/colbertv2.0",
        language_code="en",
    )
    pages = ["miyazaki", "Studio_Ghibli", "Toei_Animation"]
    my_full_corpus = [(DATA_DIR / f"{p}_wikipedia.txt").open().read() for p in pages]

    corpus_processor = CorpusProcessor(
        document_splitter_fn=llama_index_sentence_splitter
    )
    documents = corpus_processor.process_corpus(my_full_corpus, chunk_size=256)

    queries = [
        "What manga did Hayao Miyazaki write?",
        "which film made ghibli famous internationally",
        "who directed Spirited Away?",
        "when was Hikotei Jidai published?",
        "where's studio ghibli based?",
        "where is the ghibli museum?",
    ]
    pairs = []

    for query in queries:
        fake_relevant_docs = random.sample(documents, 10)
        for doc in fake_relevant_docs:
            pairs.append((query, doc))
    trainer.prepare_training_data(
        raw_data=pairs,
        data_out_path=str(tmp_path),
        all_documents=my_full_corpus,
        num_new_negatives=10,
        mine_hard_negatives=True,
    )
    try:
        with time_limit(10):
            trainer.train(
                batch_size=32,
                nbits=4,  # How many bits will the trained model use when compressing indexes
                maxsteps=1,  # Maximum steps hard stop
                use_ib_negatives=True,  # Use in-batch negative to calculate loss
                dim=128,  # How many dimensions per embedding. 128 is the default and works well.
                learning_rate=5e-6,
                # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
                doc_maxlen=256,
                # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
                use_relu=False,  # Disable ReLU -- doesn't improve performance
                warmup_steps="auto",  # Defaults to 10%
            )
            # Simply test that some of the files generated have really been made.
            assert (tmp_path / "corpus.train.colbert.tsv").exists()
    except TimeoutException as e:
        print("Timed out!")
        raise AssertionError("Timout in training") from None


if __name__ == "__main__":
    test_training()
