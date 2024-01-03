from typing import Callable, Optional, Union
from ragatouille.data.preprocessors import llama_index_sentence_splitter


class CorpusProcessor:
    def __init__(
        self,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        self.document_splitter_fn = document_splitter_fn
        self.preprocessing_fn = preprocessing_fn

    def process_corpus(
        self,
        documents: list[str],
        **splitter_kwargs,
    ) -> list[str]:
        # TODO CHECK KWARGS
        if self.document_splitter_fn is not None:
            documents = self.document_splitter_fn(documents, **splitter_kwargs)
        if self.preprocessing_fn is not None:
            if isinstance(self.preprocessing_fn, list):
                for fn in self.preprocessing_fn:
                    documents = fn(documents)
                return documents
            return self.preprocessing_fn(documents)
        return documents
