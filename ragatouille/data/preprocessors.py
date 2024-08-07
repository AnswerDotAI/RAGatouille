import re
from typing import List, Dict, Optional, Callable

try:
    try:
        from llama_index import Document
        from llama_index.text_splitter import SentenceSplitter
    except ImportError:
        from llama_index.core import Document
        from llama_index.core.text_splitter import SentenceSplitter
    has_llama_index = True
except ImportError:
    print(
        "Llamaindex is not installed, defaulting to a naive sentence splitter instead."
    )
    has_llama_index = False


def estimate_token_length(text: str) -> int:
    return int(len(text.split()) * 1.5)


def split_into_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def merge_sentences(
    sentences: List[str], chunk_size: int, chunk_overlap: int
) -> List[str]:
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_size = estimate_token_length(sentence)

        if current_chunk_size + sentence_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_size = 0
            while overlap_size < chunk_overlap and current_chunk:
                overlap_size += estimate_token_length(current_chunk[0])
                if overlap_size <= chunk_overlap:
                    current_chunk.pop(0)
            current_chunk_size = sum(estimate_token_length(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_chunk_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def naive_simple_sentence_splitter(
    documents: List[str], document_ids: List[str], chunk_size: int = 256
) -> List[Dict[str, str]]:
    chunk_overlap = min(chunk_size // 4, min(chunk_size // 2, 64))
    chunks = []

    for doc_id, doc in zip(document_ids, documents):
        sentences = split_into_sentences(doc)
        doc_chunks = merge_sentences(sentences, chunk_size, chunk_overlap)
        chunks.extend(
            [{"document_id": doc_id, "content": chunk} for chunk in doc_chunks]
        )

    return chunks


simple_sentence_splitter = naive_simple_sentence_splitter


if has_llama_index:

    def llama_index_sentence_splitter(
        documents: List[str], document_ids: List[str], chunk_size: int = 256
    ) -> List[Dict[str, str]]:
        chunk_overlap = min(chunk_size // 4, min(chunk_size // 2, 64))
        chunks = []
        node_parser = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = [Document(text=doc) for doc in documents]
        for doc_id, doc in zip(document_ids, docs):
            chunks += [
                {"document_id": doc_id, "content": node.text}
                for node in node_parser.get_nodes_from_documents([doc])
            ]
        return chunks

    simple_sentence_splitter = llama_index_sentence_splitter
