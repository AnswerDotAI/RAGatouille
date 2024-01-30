from fastapi import APIRouter
from app.api.endpoints import search, index, document_management, rerank, encode, search_encoded_docs, clear_encoded_docs

# Versioned API router
api_router = APIRouter()

# Include routers from different endpoint modules
api_router.include_router(search.router, tags=["Search"], prefix="/search")
api_router.include_router(index.router, tags=["Index"], prefix="/index")
api_router.include_router(document_management.router, tags=["Document Management"], prefix="/documents")
api_router.include_router(rerank.router, tags=["Rerank"], prefix="/rerank")
api_router.include_router(encode.router, tags=["Encode"], prefix="/encode")
api_router.include_router(search_encoded_docs.router, tags=["Search Encoded Documents"], prefix="/search-encoded")
api_router.include_router(clear_encoded_docs.router, tags=["Clear Encoded Documents"], prefix="/clear-encoded")