import warnings

_FUTURE_MIGRATION_WARNING_MESSAGE = (
    "\n********************************************************************************\n"
    "RAGatouille WARNING: Future Release Notice\n"
    "--------------------------------------------\n"
    "RAGatouille version 0.0.10 will be migrating to a PyLate backend \n"
    "instead of the current Stanford ColBERT backend.\n"
    "PyLate is a fully mature, feature-equivalent backend, that greatly facilitates compatibility.\n"
    "However, please pin version <0.0.10 if you require the Stanford ColBERT backend.\n"
    "********************************************************************************"
)

warnings.warn(
    _FUTURE_MIGRATION_WARNING_MESSAGE,
    UserWarning,
    stacklevel=2  # Ensures the warning points to the user's import line
)

__version__ = "0.0.9post2"
from .RAGPretrainedModel import RAGPretrainedModel
from .RAGTrainer import RAGTrainer

__all__ = ["RAGPretrainedModel", "RAGTrainer"]