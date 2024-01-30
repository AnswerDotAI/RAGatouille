from fastapi import APIRouter, Depends, HTTPException, status
from app.models.payloads import SearchQuery
from app.core.rag_model import get_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def search(query: SearchQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Initiating search with query: '{query.query}'. Index: '{query.index_name}', k: {query.k}")
        results = rag.search(
            query=query.query, 
            index_name=query.index_name, 
            k=query.k, 
            force_fast=query.force_fast, 
            zero_index_ranks=query.zero_index_ranks
        )
        logger.info(f"Search completed successfully for query: '{query.query}'.")
        return results
    except Exception as e:
        logger.error(f"Search operation failed for query: '{query.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Search operation failed"
        )
