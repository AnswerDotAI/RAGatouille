from fastapi import APIRouter, Depends, HTTPException, status
from app.models.payloads import ClearEncodedDocsQuery
from app.core.rag_model import get_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/clear")
async def clear_encoded_docs(query: ClearEncodedDocsQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Clearing encoded documents. Force: {query.force}")
        rag.clear_encoded_docs(force=query.force)
        logger.info("Encoded documents cleared successfully.")
        return {"message": "Encoded documents cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear encoded documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to clear encoded documents"
        )
