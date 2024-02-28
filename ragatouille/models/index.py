from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any, List, Literal, Optional, TypeAlias, Union

from colbert import IndexUpdater, Indexer, Searcher
from colbert.infra import ColBERTConfig

import torch

import srsly


IndexType: TypeAlias = Literal["FLAT", "HNSW", "PLAID"]


class ModelIndex(ABC):
    index_type: IndexType

    def __init__(
        self,
        config: ColBERTConfig,
    ) -> None:
        self.config = config

    @staticmethod
    @abstractmethod
    def construct(
        config: ColBERTConfig,
        checkpoint: str,
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> "ModelIndex":
        ...

    @staticmethod
    @abstractmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        index_config: dict[str, Any],
        config: ColBERTConfig,
        verbose: bool = True,
    ) -> "ModelIndex":
        ...

    @abstractmethod
    def build(self) -> None:
        ...

    @abstractmethod
    def search(self) -> None:
        ...

    @abstractmethod
    def batch_search(self) -> None:
        ...

    @abstractmethod
    def add(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_root: str,
        index_name: str,
        new_collection: List[str],
        verbose: bool = True,
        **kwargs,
    ) -> None:
        ...

    @abstractmethod
    def delete(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: str,
        pids_to_remove: Union[TypeVar("T"), List[TypeVar("T")]],
        verbose: bool = True,
    ) -> None:
        ...

    @abstractmethod
    def _export_config(self) -> dict[str, Any]:
        ...

    def export_metadata(self) -> dict[str, Any]:
        config = self._export_config()
        config["index_type"] = self.index_type
        return config


class FLATModelIndex(ModelIndex):
    index_type = "FLAT"


class HNSWModelIndex(ModelIndex):
    index_type = "HNSW"


class PLAIDModelIndex(ModelIndex):
    _DEFAULT_INDEX_BSIZE = 32
    index_type = "PLAID"

    def __init__(self, config: ColBERTConfig) -> None:
        super().__init__(config)

    @staticmethod
    def construct(
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> "PLAIDModelIndex":
        return PLAIDModelIndex(config).build(
            checkpoint, collection, index_name, overwrite, verbose, **kwargs
        )

    @staticmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        index_config: dict[str, Any],
        config: ColBERTConfig,
        verbose: bool = True,
    ) -> "PLAIDModelIndex":
        return PLAIDModelIndex(config)

    def build(
        self,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> "PLAIDModelIndex":
        bsize = kwargs.get("bsize", PLAIDModelIndex._DEFAULT_INDEX_BSIZE)
        assert isinstance(bsize, int)

        if torch.cuda.is_available():
            import faiss

            if not hasattr(faiss, "StandardGpuResources"):
                print(
                    "________________________________________________________________________________\n"
                    "WARNING! You have a GPU available, but only `faiss-cpu` is currently installed.\n",
                    "This means that indexing will be slow. To make use of your GPU.\n"
                    "Please install `faiss-gpu` by running:\n"
                    "pip uninstall --y faiss-cpu & pip install faiss-gpu\n",
                    "________________________________________________________________________________",
                )
                print("Will continue with CPU indexing in 5 seconds...")
                time.sleep(5)

        nbits = 2
        if len(collection) < 5000:
            nbits = 8
        elif len(collection) < 10000:
            nbits = 4
        self.config = ColBERTConfig.from_existing(
            self.config, ColBERTConfig(nbits=nbits, index_bsize=bsize)
        )

        if len(collection) > 100000:
            self.config.kmeans_niters = 4
        elif len(collection) > 50000:
            self.config.kmeans_niters = 10
        else:
            self.config.kmeans_niters = 20

        # Instruct colbert-ai to disable forking if nranks == 1
        self.config.avoid_fork_if_possible = True
        indexer = Indexer(
            checkpoint=checkpoint,
            config=self.config,
            verbose=verbose,
        )
        indexer.configure(avoid_fork_if_possible=True)
        indexer.index(name=index_name, collection=collection, overwrite=overwrite)
        return self

    def search(self) -> None:
        raise NotImplementedError()

    def batch_search(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def _should_rebuild(current_len: int, new_doc_len: int) -> bool:
        """
        Heuristic to determine if it is more efficient to rebuild the index instead of updating it.
        """
        return current_len + new_doc_len < 5000 or new_doc_len > current_len * 0.05

    def add(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_root: str,
        index_name: str,
        new_collection: List[str],
        verbose: bool = True,
        **kwargs,
    ) -> None:
        self.config = config

        bsize = kwargs.get("bsize", PLAIDModelIndex._DEFAULT_INDEX_BSIZE)
        assert isinstance(bsize, int)

        searcher = Searcher(
            checkpoint=checkpoint,
            config=None,
            collection=collection,
            index=index_name,
            index_root=index_root,
            verbose=verbose,
        )

        if PLAIDModelIndex._should_rebuild(
            len(searcher.collection), len(new_collection)
        ):
            self.build(
                checkpoint=checkpoint,
                collection=collection + new_collection,
                index_name=index_name,
                overwrite="force_silent_overwrite",
                verbose=verbose,
                **kwargs,
            )
        else:
            if self.config.index_bsize != bsize:  # Update bsize if it's different
                self.config.index_bsize = bsize

            updater = IndexUpdater(
                config=self.config, searcher=searcher, checkpoint=checkpoint
            )
            updater.add(new_collection)
            updater.persist_to_disk()

    def delete(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: str,
        pids_to_remove: Union[TypeVar("T"), List[TypeVar("T")]],
        verbose: bool = True,
    ) -> None:
        self.config = config

        # Initialize the searcher and updater
        searcher = Searcher(
            checkpoint=checkpoint,
            config=None,
            collection=collection,
            index=index_name,
            verbose=verbose,
        )
        updater = IndexUpdater(config=config, searcher=searcher, checkpoint=checkpoint)

        updater.remove(pids_to_remove)
        updater.persist_to_disk()

    def _export_config(self) -> dict[str, Any]:
        return {}


class ModelIndexFactory:
    _MODEL_INDEX_BY_NAME = {
        "FLAT": FLATModelIndex,
        "HNSW": HNSWModelIndex,
        "PLAID": PLAIDModelIndex,
    }

    @staticmethod
    def _raise_if_invalid_index_type(index_type: str) -> IndexType:
        if index_type not in ["FLAT", "HNSW", "PLAID"]:
            raise ValueError(
                f"Unsupported index_type `{index_type}`; it must be one of 'FLAT', 'HNSW', OR 'PLAID'"
            )
        return index_type  # type: ignore

    @staticmethod
    def construct(
        index_type: Union[Literal["auto"], IndexType],
        config: ColBERTConfig,
        checkpoint: str,
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> ModelIndex:
        # Automatically choose the appropriate index for the desired "workload".
        if index_type == "auto":
            # NOTE: For now only PLAID indexes are supported.
            index_type = "PLAID"
        return ModelIndexFactory._MODEL_INDEX_BY_NAME[
            ModelIndexFactory._raise_if_invalid_index_type(index_type)
        ].construct(
            config, checkpoint, collection, index_name, overwrite, verbose, **kwargs
        )

    @staticmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        config: ColBERTConfig,
        verbose: bool = True,
    ) -> ModelIndex:
        metadata = srsly.read_json(index_path + "/metadata.json")
        try:
            index_config = metadata["RAGatouille"]["index_config"]  # type: ignore
        except KeyError:
            if verbose:
                print(
                    f"Constructing default index configuration for index `{index_name}` as it does not contain RAGatouille specific metadata."
                )
            index_config = {
                "index_type": "PLAID",
                "index_name": index_name,
            }
        index_name = (
            index_name if index_name is not None else index_config["index_name"]  # type: ignore
        )
        return ModelIndexFactory._MODEL_INDEX_BY_NAME[
            ModelIndexFactory._raise_if_invalid_index_type(index_config["index_type"])  # type: ignore
        ].load_from_file(index_path, index_name, index_config, config, verbose)
