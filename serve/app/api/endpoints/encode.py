from fastapi import APIRouter, Depends, HTTPException, status
from app.models.payloads import EncodeQuery
from app.core.rag_model import get_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def encode(query: EncodeQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Starting document encoding. Batch size: {query.bsize}, Verbose: {query.verbose}")
        rag.encode(
            documents=query.documents,
            bsize=query.bsize,
            document_metadatas=query.document_metadatas,
            verbose=query.verbose,
            max_document_length=query.max_document_length
        )
        logger.info(f"Documents encoded successfully. Total documents: {len(query.documents)}")
        return {"message": "Documents encoded successfully"}
    except Exception as e:
        logger.error(f"Failed to encode documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to encode documents"
        )
