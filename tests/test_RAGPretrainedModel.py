import os

import pytest
import srsly

from ragatouille import RAGPretrainedModel

collection = [
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


@pytest.fixture(scope="session")
def RAG_from_pretrained_model(persistent_temp_index_root):
    return RAGPretrainedModel.from_pretrained(
        "colbert-ir/colbertv2.0", index_root=str(persistent_temp_index_root)
    )


@pytest.fixture(scope="session")
def index_path_fixture(persistent_temp_index_root, index_creation_inputs):
    index_path = os.path.join(
        str(persistent_temp_index_root),
        "colbert",
        "indexes",
        index_creation_inputs["index_name"],
    )
    return str(index_path)


@pytest.fixture(scope="session")
def collection_path_fixture(index_path_fixture):
    collection_path = os.path.join(index_path_fixture, "collection.json")
    return str(collection_path)


@pytest.fixture(scope="session")
def document_metadata_path_fixture(index_path_fixture):
    document_metadata_path = os.path.join(index_path_fixture, "docid_metadata_map.json")
    return str(document_metadata_path)


@pytest.fixture(scope="session")
def pid_docid_map_path_fixture(index_path_fixture):
    pid_docid_map_path = os.path.join(index_path_fixture, "pid_docid_map.json")
    return str(pid_docid_map_path)


@pytest.fixture(
    scope="session",
    params=[
        {
            "collection": collection,
            "index_name": "no_optional_args",
            "split_documents": False,
        },
        {
            "collection": collection,
            "document_ids": document_ids,
            "index_name": "with_docid",
            "split_documents": False,
        },
        {
            "collection": collection,
            "document_metadatas": document_metadatas,
            "index_name": "with_metadata",
            "split_documents": False,
        },
        {
            "collection": collection,
            "index_name": "with_split",
            "split_documents": True,
        },
        {
            "collection": collection,
            "document_ids": document_ids,
            "document_metadatas": document_metadatas,
            "index_name": "with_docid_metadata",
            "split_documents": False,
        },
        {
            "collection": collection,
            "document_ids": document_ids,
            "index_name": "with_docid_split",
            "split_documents": True,
        },
        {
            "collection": collection,
            "document_metadatas": document_metadatas,
            "index_name": "with_metadata_split",
            "split_documents": True,
        },
        {
            "collection": collection,
            "document_ids": document_ids,
            "document_metadatas": document_metadatas,
            "index_name": "with_docid_metadata_split",
            "split_documents": True,
        },
    ],
    ids=[
        "No optional arguments",
        "With document IDs",
        "With metadata",
        "With document splitting",
        "With document IDs and metadata",
        "With document IDs and splitting",
        "With metadata and splitting",
        "With document IDs, metadata, and splitting",
    ],
)
def index_creation_inputs(request):
    params = request.param
    return params


@pytest.fixture(scope="session")
def create_index(RAG_from_pretrained_model, index_creation_inputs):
    index_path = RAG_from_pretrained_model.index(**index_creation_inputs)
    return index_path


def test_index_creation(create_index):
    assert os.path.exists(create_index) == True


@pytest.fixture(scope="session", autouse=True)
def add_docids_to_index_inputs(
    create_index, index_creation_inputs, pid_docid_map_path_fixture
):
    if "document_ids" not in index_creation_inputs:
        pid_docid_map_data = srsly.read_json(pid_docid_map_path_fixture)
        seen_ids = set()
        index_creation_inputs["document_ids"] = [
            x
            for x in list(pid_docid_map_data.values())
            if not (x in seen_ids or seen_ids.add(x))
        ]


def test_collection_creation(collection_path_fixture):
    assert os.path.exists(collection_path_fixture) == True
    collection_data = srsly.read_json(collection_path_fixture)
    assert isinstance(
        collection_data, list
    ), "The collection.json file should contain a list."


def test_pid_docid_map_creation(pid_docid_map_path_fixture):
    assert os.path.exists(pid_docid_map_path_fixture) == True
    # TODO check pid_docid_map_data
    pid_docid_map_data = srsly.read_json(pid_docid_map_path_fixture)
    assert isinstance(
        pid_docid_map_data, dict
    ), "The pid_docid_map.json file should contain a dictionary."


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
            ), f"The metadata for document_id {doc_id} should match the provided metadata."
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


# def test_return_entire_document(index_creation_inputs, index_path_fixture):
#     if index_creation_inputs["split_documents"] == True:
#         RAG = RAGPretrainedModel.from_index(index_path_fixture)
#         results = RAG.search(
#             "when was miyazaki born",
#             index_name=index_creation_inputs["index_name"],
#             return_entire_document=True,
#         )
#         for result in results:
#             assert (
#                 "entire_document" in result
#             ), "The full document should be returned in the results."
#             doc_id = result["document_id"]
#             expected_document = index_creation_inputs["collection"][
#                 index_creation_inputs["document_ids"].index(doc_id)
#             ]
#             assert (
#                 result["entire_document"] == expected_document
#             ), f"The document for document_id {doc_id} should match the provided document."
#     else:
#         assert True, "This test is only relevant for split documents."


def test_delete_from_index(
    index_creation_inputs,
    pid_docid_map_path_fixture,
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
    pid_docid_map_data = srsly.read_json(pid_docid_map_path_fixture)
    updated_document_ids = set(list(pid_docid_map_data.values()))
    assert (
        deleted_doc_id not in updated_document_ids
    ), "Deleted document ID should not be in the collection."
    assert original_doc_ids - updated_document_ids == {
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


def test_add_to_index(
    index_creation_inputs,
    document_metadata_path_fixture,
    pid_docid_map_path_fixture,
    index_path_fixture,
):
    RAG = RAGPretrainedModel.from_index(index_path_fixture)
    new_doc_ids = ["mononoke", "sabaku_no_tami"]
    new_docs = [
        "Princess Mononoke (Japanese: もののけ姫, Hepburn: Mononoke-hime) is a 1997 Japanese animated epic historical fantasy film written and directed by Hayao Miyazaki and animated by Studio Ghibli for Tokuma Shoten, Nippon Television Network and Dentsu. The film stars the voices of Yōji Matsuda, Yuriko Ishida, Yūko Tanaka, Kaoru Kobayashi, Masahiko Nishimura, Tsunehiko Kamijo, Akihiro Miwa, Mitsuko Mori, and Hisaya Morishige.\nPrincess Mononoke is set in the late Muromachi period of Japan (approximately 1336 to 1573 AD) and includes fantasy elements. The story follows a young Emishi prince named Ashitaka, and his involvement in a struggle between the gods (kami) of a forest and the humans who consume its resources. The film deals with themes of Shinto and environmentalism.\nThe film was released in Japan on July 12, 1997, by Toho, and in the United States on October 29, 1999. This was the first Studio Ghibli film in the United States to be rated PG-13 by the MPA. It was a critical and commercial blockbuster, becoming the highest-grossing film in Japan of 1997, and also held Japan's box office record for domestic films until 2001's Spirited Away, another Miyazaki film. It was dubbed into English with a script by Neil Gaiman and initially distributed in North America by Miramax, where it sold well on home media despite not performing strongly at the box office. The film greatly increased Ghibli's popularity and influence outside Japan.",
        "People of the Desert (砂漠の民, Sabaku no Tami, translated on the cover as The People of Desert), or The Desert Tribe, is a comic strip written and illustrated by Hayao Miyazaki. It was serialized, under the pseudonym Akitsu Saburō (秋津三朗), and ran in Boys and Girls Newspaper (少年少女新聞, Shōnen Shōjo Shinbun) between September 12, 1969, and March 15, 1970.\n\n\n== Story ==\nThe story is set in the distant past, on the fictionalised desert plains of Central Asia. Part of the story takes place in the fortified city named Pejite (ペジテ). The story follows the exploits of the main character, Tem (テム, Temu), a shepherd boy of the fictional Sokut (ソクート, Sokūto) tribe, as he tries to evade the mounted militia of the nomadic Kittāru (キッタール) tribe. In order to restore peace to the realm, Tem rallies his remaining compatriots and rebels against the Kittāru's attempts to gain control of the Sokut territory and enslave its inhabitants through military force.\n\n\n== Background, publication and influences ==\nMiyazaki initially wanted to become a manga artist but started his professional career as an animator for Toei Animation in 1963. Here he worked on animated television series and animated feature-length films for theatrical release. He never abandoned his childhood dream of becoming a manga artist completely, however, and his professional debut as a manga creator came in 1969 with the publication of his manga interpretation of Puss 'n Boots, which was serialized in 12 weekly instalments in the Sunday edition of Tokyo Shimbun, from January to March 1969. Printed in colour and created for promotional purposes in conjunction with his work on Toei's animated film of the same title, directed by Kimio Yabuki.\nIn 1969 pseudonymous serialization also started of Miyazaki's original manga People of the Desert (砂漠の民, Sabaku no Tami). This strip was created in the style of illustrated stories (絵物語, emonogatari) he read in boys' magazines and tankōbon volumes while growing up, such as Soji Yamakawa's Shōnen Ōja (少年王者) and in particular Tetsuji Fukushima's Evil Lord of the Desert (沙漠の魔王, Sabaku no Maō). Miyazaki's People of the Desert is a continuation of that tradition. In People of the Desert expository text is presented separately from the monochrome artwork but Miyazaki progressively used additional text balloons inside the panels for dialogue.\nPeople of the Desert was serialized in 26 weekly instalments which were printed in Boys and Girls Newspaper (少年少女新聞, Shōnen shōjo shinbun), a publication of the Japanese Communist Party, between September 12, 1969 (issue 28) and March 15, 1970 (issue 53). The strip was published under the pseudonym Akitsu Saburō (秋津三朗).\nThe strip has been identified as a precursor for Miyazaki's manga Nausicaä of the Valley of the Wind (1982–1995) and the one-off graphic novel Shuna's Journey (1983), published by Tokuma Shoten.",
    ]
    new_doc_metadata = [
        {"entity": "film", "source": "wikipedia"},
        {"entity": "manga", "source": "wikipedia"},
    ]
    RAG.add_to_index(
        new_collection=new_docs,
        new_document_ids=new_doc_ids,
        new_document_metadatas=new_doc_metadata,
        index_name=index_creation_inputs["index_name"],
    )
    pid_docid_map_data = srsly.read_json(pid_docid_map_path_fixture)
    document_ids = set(list(pid_docid_map_data.values()))

    document_metadata_dict = srsly.read_json(document_metadata_path_fixture)
    for new_doc_id in new_doc_ids:
        assert (
            new_doc_id in document_ids
        ), f"New document ID {new_doc_id} should be in the pid_docid_map."
        assert (
            new_doc_id in document_metadata_dict
        ), f"New document ID {new_doc_id} should be in the document metadata."
