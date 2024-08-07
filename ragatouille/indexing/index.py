# import time
# from abc import ABC, abstractmethod
# from copy import deepcopy
# from pathlib import Path
# from typing import Any, List, Literal, Optional, TypeVar, Union

# import srsly
# import torch
# from colbert import Indexer, IndexUpdater, Searcher
# from colbert.indexing.collection_indexer import CollectionIndexer
# from colbert.infra import ColBERTConfig

# from ragatouille.models import torch_kmeans

# IndexType = Literal["FLAT", "HNSW", "PLAID"]


# class FLATModelIndex(ModelIndex):
#     index_type = "FLAT"

