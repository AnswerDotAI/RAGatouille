import pytest
from ragatouille import RAGPretrainedModel
import os
import srsly


documents = [
    "Hayao Miyazaki (宮崎 駿 or 宮﨑 駿, Miyazaki Hayao, [mijaꜜzaki hajao]; born January 5, 1941) is a Japanese animator, filmmaker, and manga artist. A co-founder of Studio Ghibli, he has attained international acclaim as a masterful storyteller and creator of Japanese animated feature films, and is widely regarded as one of the most accomplished filmmakers in the history of animation.\nBorn in Tokyo City in the Empire of Japan, Miyazaki expressed interest in manga and animation from an early age, and he joined Toei Animation in 1963. During his early years at Toei Animation he worked as an in-between artist and later collaborated with director Isao Takahata. Notable films to which Miyazaki contributed at Toei include Doggie March and Gulliver's Travels Beyond the Moon. He provided key animation to other films at Toei, such as Puss in Boots and Animal Treasure Island, before moving to A-Pro in 1971, where he co-directed Lupin the Third Part I alongside Takahata. After moving to Zuiyō Eizō (later known as Nippon Animation) in 1973, Miyazaki worked as an animator on World Masterpiece Theater, and directed the television series Future Boy Conan (1978). He joined Tokyo Movie Shinsha in 1979 to direct his first feature film The Castle of Cagliostro as well as the television series Sherlock Hound. In the same period, he also began writing and illustrating the manga Nausicaä of the Valley of the Wind (1982–1994), and he also directed the 1984 film adaptation produced by Topcraft.\nMiyazaki co-founded Studio Ghibli in 1985. He directed numerous films with Ghibli, including Laputa: Castle in the Sky (1986), My Neighbor Totoro (1988), Kiki's Delivery Service (1989), and Porco Rosso (1992). The films were met with critical and commercial success in Japan. Miyazaki's film Princess Mononoke was the first animated film ever to win the Japan Academy Prize for Picture of the Year, and briefly became the highest-grossing film in Japan following its release in 1997; its distribution to the Western world greatly increased Ghibli's popularity and influence outside Japan. His 2001 film Spirited Away became the highest-grossing film in Japanese history, winning the Academy Award for Best Animated Feature, and is frequently ranked among the greatest films of the 21st century. Miyazaki's later films—Howl's Moving Castle (2004), Ponyo (2008), and The Wind Rises (2013)—also enjoyed critical and commercial success.",
    "Studio Ghibli, Inc. (Japanese: 株式会社スタジオジブリ, Hepburn: Kabushiki gaisha Sutajio Jiburi) is a Japanese animation studio based in Koganei, Tokyo. It has a strong presence in the animation industry and has expanded its portfolio to include various media formats, such as short subjects, television commercials, and two television films. Their work has been well-received by audiences and recognized with numerous awards. Their mascot and most recognizable symbol, the character Totoro from the 1988 film My Neighbor Totoro, is a giant spirit inspired by raccoon dogs (tanuki) and cats (neko). Among the studio's highest-grossing films are Spirited Away (2001), Howl's Moving Castle (2004), and Ponyo (2008). Studio Ghibli was founded on June 15, 1985, by the directors Hayao Miyazaki and Isao Takahata and producer Toshio Suzuki, after acquiring Topcraft's assets. The studio has also collaborated with video game studios on the visual development of several games.Five of the studio's films are among the ten highest-grossing anime feature films made in Japan. Spirited Away is second, grossing 31.68 billion yen in Japan and over US$380 million worldwide, and Princess Mononoke is fourth, grossing 20.18 billion yen. Three of their films have won the Animage Grand Prix award, four have won the Japan Academy Prize for Animation of the Year, and five have received Academy Award nominations. Spirited Away won the 2002 Golden Bear and the 2003 Academy Award for Best Animated Feature.On August 3, 2014, Studio Ghibli temporarily suspended production following Miyazaki's retirement.",
]

document_ids = ["miyazaki", "ghibli"]

document_metadatas = [
    {"entity": "person", "source": "wikipedia"},
    {"entity": "organisation", "source": "wikipedia"},
]


@pytest.fixture(scope="session")
def persistent_temp_index_root(tmp_path_factory):
    return tmp_path_factory.mktemp("temp_test_indexes")


@pytest.fixture
def RAG_from_pretrained_model(persistent_temp_index_root):
    return RAGPretrainedModel.from_pretrained(
        "colbert-ir/colbertv2.0", index_root=str(persistent_temp_index_root)
    )


@pytest.fixture
def index_path_fixture(persistent_temp_index_root, index_creation_inputs):
    index_path = os.path.join(
        str(persistent_temp_index_root),
        "colbert",
        "indexes",
        index_creation_inputs["index_name"],
    )
    return str(index_path)


@pytest.fixture
def collection_path_fixture(index_path_fixture):
    collection_path = os.path.join(index_path_fixture, "collection.json")
    return str(collection_path)


@pytest.fixture
def document_metadata_path_fixture(index_path_fixture):
    document_metadata_path = os.path.join(index_path_fixture, "document_metadata.json")
    return str(document_metadata_path)


@pytest.fixture(
    params=[
        {
            "documents": documents,
            "document_ids": document_ids,
            "index_name": "no_split_no_metadata",
            "split_documents": False,
        },
        {
            "documents": documents,
            "document_ids": document_ids,
            "index_name": "split_no_metadata",
            "split_documents": True,
        },
        {
            "documents": documents,
            "document_ids": document_ids,
            "document_metadatas": document_metadatas,
            "index_name": "split_with_metadata",
            "split_documents": True,
        },
        {
            "documents": documents,
            "document_ids": document_ids,
            "document_metadatas": document_metadatas,
            "index_name": "no_split_with_metadata",
            "split_documents": False,
        },
    ],
    ids=[
        "No document splitting, no metadata",
        "Document splitting, no metadata",
        "Document splitting, with metadata",
        "No document splitting, with metadata",
    ],
)
def index_creation_inputs(request):
    return request.param


def test_index_creation(RAG_from_pretrained_model, index_creation_inputs):
    RAG = RAG_from_pretrained_model
    index_path = RAG.index(**index_creation_inputs)
    assert os.path.exists(index_path) == True


def test_document_id_in_collection(index_creation_inputs, collection_path_fixture):
    assert os.path.exists(collection_path_fixture) == True
    collection_data = srsly.read_json(collection_path_fixture)
    assert isinstance(
        collection_data, list
    ), "The collection.json file should contain a list."

    collected_document_ids = [
        item["document_id"] for item in collection_data if "document_id" in item
    ]

    for item in collection_data:
        assert (
            "document_id" in item
        ), "Each item in collection.json should have a 'document_id' key."

    assert (
        set(collected_document_ids) == set(index_creation_inputs["document_ids"])
    ), "All document_ids provided for index creation should be present in the collection.json."


def test_document_metadata_creation(
    index_creation_inputs, document_metadata_path_fixture
):
    if "document_metadatas" in index_creation_inputs:
        assert os.path.exists(document_metadata_path_fixture) == True
        document_metadata_dict = srsly.read_json(document_metadata_path_fixture)
        assert (
            set(document_metadata_dict.keys())
            == set(index_creation_inputs["document_ids"])
        ), "The keys in document_metadata.json should match the document_ids provided for index creation."
        for doc_id, metadata in document_metadata_dict.items():
            assert (
                metadata
                == index_creation_inputs["document_metadatas"][
                    index_creation_inputs["document_ids"].index(doc_id)
                ]
            ), "The metadata for document_id {} should match the provided metadata.".format(
                doc_id
            )
    else:
        assert os.path.exists(document_metadata_path_fixture) == False


def test_document_metadata_returned_in_search_results(
    index_creation_inputs, index_path_fixture
):
    RAG = RAGPretrainedModel.from_index(index_path_fixture)
    results = RAG.search(
        "when was miyazaki born", index_name=index_creation_inputs["index_name"]
    )

    if "document_metadatas" in index_creation_inputs:
        for result in results:
            assert (
                "document_metadata" in result
            ), "The metadata should be returned in the results."
            doc_id = result["document_id"]
            expected_metadata = index_creation_inputs["document_metadatas"][
                index_creation_inputs["document_ids"].index(doc_id)
            ]
            assert (
                result["document_metadata"] == expected_metadata
            ), f"The metadata for document_id {doc_id} should match the provided metadata."
    else:
        for result in results:
            assert (
                "metadata" not in result
            ), "The metadata should not be returned in the results."


def test_delete_from_index(
    index_creation_inputs,
    collection_path_fixture,
    document_metadata_path_fixture,
    index_path_fixture,
):
    RAG = RAGPretrainedModel.from_index(index_path_fixture)
    deleted_doc_id = index_creation_inputs["document_ids"][0]
    original_doc_ids = set(index_creation_inputs["document_ids"])
    RAG.delete_from_index(
        index_name=index_creation_inputs["index_name"],
        document_ids=[deleted_doc_id],
    )
    collection_data = srsly.read_json(collection_path_fixture)
    collection_data_ids = set([item["document_id"] for item in collection_data])
    assert (
        deleted_doc_id not in collection_data_ids
    ), "Deleted document ID should not be in the collection."
    assert original_doc_ids - collection_data_ids == {
        deleted_doc_id
    }, "Only the deleted document ID should be missing from the collection."
    if "document_metadatas" in index_creation_inputs:
        document_metadata_dict = srsly.read_json(document_metadata_path_fixture)
        assert (
            deleted_doc_id not in document_metadata_dict
        ), "Deleted document ID should not be in the document metadata."
        assert original_doc_ids - set(document_metadata_dict.keys()) == {
            deleted_doc_id
        }, "Only the deleted document ID should be missing from the document metadata."

@pytest.mark.skip(reason="Not implemented yet.")
def test_add_to_index(
    index_creation_inputs,
    collection_path_fixture,
    document_metadata_path_fixture,
    index_path_fixture,
):
    RAG = RAGPretrainedModel.from_index(index_path_fixture)
    new_doc_id = "new_doc_id"
    new_doc = "This is a new document."
    RAG.add_to_index(
        new_documents=[new_doc],
        new_metadata=[{"entity": "person", "source": "wikipedia"}],
        index_name=index_creation_inputs["index_name"],
    )
    collection_data = srsly.read_json(collection_path_fixture)
    collection_data_ids = set([item["document_id"] for item in collection_data])
    assert (
        new_doc_id in collection_data_ids
    ), "New document ID should be in the collection."

    document_metadata_dict = srsly.read_json(document_metadata_path_fixture)
    assert (
        new_doc_id in document_metadata_dict
    ), "New document ID should be in the document metadata."
