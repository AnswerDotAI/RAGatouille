from fastapi import APIRouter, Depends, HTTPException, status
from app.models.payloads import SearchEncodedDocsQuery
from app.core.rag_model import get_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def search_encoded_docs(query: SearchEncodedDocsQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Starting search in encoded documents. Query: '{query.query}'")
        results = rag.search_encoded_docs(
            query=query.query,
            k=query.k,
            bsize=query.bsize
        )
        logger.info(f"Search in encoded documents completed successfully. Query: '{query.query}'")
        return results
    except Exception as e:
        logger.error(f"Search in encoded documents failed for query: '{query.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search in encoded documents failed"
        )
