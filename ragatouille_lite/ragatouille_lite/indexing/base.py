from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Literal, Optional, TypeVar, Union

try:
    import ragatouille
    from colbert.infra import ColBERTConfig
except:
    pass


IndexType = Literal["FLAT", "HNSW", "PLAID"]


class ModelIndex(ABC):
    index_type: IndexType

    def __init__(
        self,
        config: Any,
    ) -> None:
        self.config = config
        self.collection = None
        self.pid_docid_map = None
        self.docid_pid_map = None
        self.docid_metadata_map = None

    @staticmethod
    @abstractmethod
    def construct(
        config: Any,
        checkpoint: str,
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> "ModelIndex": ...

    @staticmethod
    @abstractmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        index_config: dict[str, Any],
        config: Any,
        verbose: bool = True,
    ) -> "ModelIndex": ...

    @abstractmethod
    def build(
        self,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        config: Any,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional[str],
        base_model_max_tokens: int,
        query: Union[str, list[str]],
        k: int = 10,
        pids: Optional[List[int]] = None,
        force_reload: bool = False,
        **kwargs,
    ) -> list[tuple[list, list, list]]: ...

    @abstractmethod
    def _search(self, query: str, k: int, pids: Optional[List[int]] = None): ...

    @abstractmethod
    def _batch_search(self, query: list[str], k: int): ...

    @abstractmethod
    def add(
        self,
        config: Any,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_root: str,
        index_name: str,
        new_collection: List[str],
        verbose: bool = True,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def delete(
        self,
        config: Any,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: str,
        pids_to_remove: Union[TypeVar("T"), List[TypeVar("T")]],
        verbose: bool = True,
    ) -> None: ...

    @abstractmethod
    def save(self, index_name: str, index_root: str, verbose: bool = True) -> None: ...

    @abstractmethod
    def load(self, index_name: str, verbose: bool = True) -> None: ...

    @abstractmethod
    def load_from_hf_hub(
        self, repo_id: str, index_name: str, verbose: bool = True
    ) -> None: ...

    @abstractmethod
    def push_to_hf_hub(
        self, repo_id: str, index_name: str, verbose: bool = True
    ) -> None: ...

    @abstractmethod
    def _export_config(self) -> dict[str, Any]: ...

    def export_metadata(self) -> dict[str, Any]:
        config = self._export_config()
        config["index_type"] = self.index_type
        return config
