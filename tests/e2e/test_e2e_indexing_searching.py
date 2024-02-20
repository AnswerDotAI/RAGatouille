import pytest
import srsly

from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page


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
    results = RAG.search(query="What animation studio did Miyazaki found?", k=k)
    assert len(results) == k
    assert (
        "In April 1984, Miyazaki opened his own office in Suginami Ward"
        in results[0]["content"]
    )
    assert (
        "Hayao Miyazaki (宮崎 駿 or 宮﨑 駿, Miyazaki Hayao, [mijaꜜzaki hajao]; born January 5, 1941)"  # noqa
        in results[1]["content"]
    )
    assert (
        'Glen Keane said Miyazaki is a "huge influence" on Walt Disney Animation Studios and has been'  # noqa
        in results[2]["content"]
    )

    all_results = RAG.search(
        query=["What animation studio did Miyazaki found?", "Miyazaki son name"], k=k
    )
    assert (
        "In April 1984, Miyazaki opened his own office in Suginami Ward"
        in all_results[0][0]["content"]
    )
    assert (
        "Hayao Miyazaki (宮崎 駿 or 宮﨑 駿, Miyazaki Hayao, [mijaꜜzaki hajao]; born January 5, 1941)"  # noqa
        in all_results[0][1]["content"]
    )
    assert (
        'Glen Keane said Miyazaki is a "huge influence" on Walt Disney Animation Studios and has been'  # noqa
        in all_results[0][2]["content"]
    )
    assert (
        "== Early life ==\nHayao Miyazaki was born on January 5, 1941"
        in all_results[1][0]["content"]  # noqa
    )
    assert (
        "Directed by Isao Takahata, with whom Miyazaki would continue to collaborate for the remainder of his career"  # noqa
        in all_results[1][1]["content"]
    )
    actual = all_results[1][2]["content"]
    assert (
        "Specific works that have influenced Miyazaki include Animal Farm (1945)"
        in actual
        or "She met with Suzuki" in actual
    )
    print(all_results)


@pytest.mark.skip(reason="experimental feature.")
def test_basic_CRUD_addition():
    old_collection = srsly.read_json(
        ".ragatouille/colbert/indexes/Miyazaki/collection.json"
    )
    old_collection_len = len(old_collection)
    path_to_index = ".ragatouille/colbert/indexes/Miyazaki/"
    RAG = RAGPretrainedModel.from_index(path_to_index)

    new_documents = get_wikipedia_page("Studio_Ghibli")

    RAG.add_to_index([new_documents])
    new_collection = srsly.read_json(
        ".ragatouille/colbert/indexes/Miyazaki/collection.json"
    )
    assert len(new_collection) > old_collection_len
    assert len(new_collection) == 140
