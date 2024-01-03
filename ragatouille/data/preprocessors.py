from llama_index import Document
from llama_index.text_splitter import SentenceSplitter


def llama_index_sentence_splitter(documents: list[str], chunk_size=256):
    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
    chunks = []
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = [[Document(text=doc)] for doc in documents]
    for doc in docs:
        chunks += [node.text for node in node_parser(doc)]
    return chunks
