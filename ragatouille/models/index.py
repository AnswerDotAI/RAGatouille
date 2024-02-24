from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any, List, Literal, Optional, TypeAlias, Union

from colbert import Indexer
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
    def load_from_file(pretrained_model_path: Path) -> "ModelIndex":
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
    def add(self) -> None:
        ...

    @abstractmethod
    def delete(self) -> None:
        ...

    @abstractmethod
    def export(self) -> Optional[dict[str, Any]]:
        ...


class FLATModelIndex(ModelIndex):
    index_type = "FLAT"


class HNSWModelIndex(ModelIndex):
    index_type = "HNSW"


class PLAIDModelIndex(ModelIndex):
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
        bsize = kwargs.get("bsize", 32)
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
        config = ColBERTConfig.from_existing(
            config, ColBERTConfig(nbits=nbits, index_bsize=bsize)
        )

        if len(collection) > 100000:
            config.kmeans_niters = 4
        elif len(collection) > 50000:
            config.kmeans_niters = 10
        else:
            config.kmeans_niters = 20

        # Instruct colbert-ai to disable forking if nranks == 1
        config.avoid_fork_if_possible = True
        indexer = Indexer(
            checkpoint=checkpoint,
            config=config,
            verbose=verbose,
        )
        indexer.configure(avoid_fork_if_possible=True)
        indexer.index(name=index_name, collection=collection, overwrite=overwrite)
        return PLAIDModelIndex(config)

    @staticmethod
    def load_from_file(pretrained_model_path: Path) -> "PLAIDModelIndex":
        raise NotImplementedError()

    def build(self) -> None:
        raise NotImplementedError()

    def search(self) -> None:
        raise NotImplementedError()

    def batch_search(self) -> None:
        raise NotImplementedError()

    def add(self) -> None:
        raise NotImplementedError()

    def delete(self) -> None:
        raise NotImplementedError()

    def export(self) -> Optional[dict[str, Any]]:
        raise NotImplementedError()


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
    def _file_index_type(pretrained_model_path: Path) -> IndexType:
        try:
            index_type = srsly.read_json(str(pretrained_model_path / "metadata.json"))[
                "index_type"
            ]
            assert isinstance(index_type, str)
        except KeyError:
            index_type = "PLAID"
        return ModelIndexFactory._raise_if_invalid_index_type(index_type)

    @staticmethod
    def load_from_file(pretrained_model_path: Path) -> ModelIndex:
        index_type = ModelIndexFactory._file_index_type(pretrained_model_path)
        return ModelIndexFactory._MODEL_INDEX_BY_NAME[index_type].load_from_file(
            pretrained_model_path
        )
