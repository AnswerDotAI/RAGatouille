import srsly

from ragatouille import RAGPretrainedModel


def test_indexing():
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    with open("tests/data/miyazaki_wikipedia.txt", "r") as f:
        full_document = f.read()
    RAG.index(
        collection=[full_document],
        index_name="Miyazaki",
        max_document_length=180,
        split_documents=True,
    )
    # ensure collection is stored to disk
    collection = srsly.read_json(
        ".ragatouille/colbert/indexes/Miyazaki/collection.json"
    )
    assert len(collection) > 1


def test_search():
    RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/Miyazaki/")
    k = 3  # How many documents you want to retrieve, defaults to 10, we set it to 3 here for readability
