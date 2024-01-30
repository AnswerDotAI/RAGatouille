from fastapi import Depends
from app.core.config import settings
from ragatouille import RAGPretrainedModel
import logging

# Set up a logger
logger = logging.getLogger(__name__)

# Singleton pattern to avoid loading the model multiple times
class RAGModel:
    model_instance: RAGPretrainedModel = None

    @classmethod
    def get_instance(cls) -> RAGPretrainedModel:
        if cls.model_instance is None:
            logger.info("Initializing RAGPretrainedModel instance...")
            # Initialize and configure RAGPretrainedModel
            cls.model_instance = RAGPretrainedModel.from_pretrained(settings.MODEL_NAME)
            logger.info("RAGPretrainedModel instance created successfully.")
        else:
            logger.debug("RAGPretrainedModel instance already exists, reusing the existing instance.")
        return cls.model_instance

def get_rag_model() -> RAGPretrainedModel:
    return RAGModel.get_instance()
