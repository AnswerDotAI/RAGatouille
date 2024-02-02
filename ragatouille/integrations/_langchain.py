from typing import Any, List, Optional, Sequence

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RAGatouilleLangChainRetriever(BaseRetriever):
    model: Any
    kwargs: dict = {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa
    ) -> List[Document]:
        """Get documents relevant to a query."""
        docs = self.model.search(query, **self.kwargs)
        return [
            Document(
                page_content=doc["content"], metadata=doc.get("document_metadata", {})
            )
            for doc in docs
        ]


class RAGatouilleLangChainCompressor(BaseDocumentCompressor):
    model: Any
    kwargs: dict = {}
    k: int = 5

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,  # noqa
        **kwargs,
    ) -> Any:
        """Rerank a list of documents relevant to a query."""
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.model.rerank(
            query=query,
            documents=_docs,
            k=kwargs.get("k", self.k),
            **self.kwargs,
        )
        final_results = []
        for r in results:
            doc = doc_list[r["result_index"]]
            doc.metadata["relevance_score"] = r["score"]
            final_results.append(doc)
        return final_results
