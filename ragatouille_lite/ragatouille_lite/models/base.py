from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, List
import srsly


class LateInteractionModel(ABC):
    @abstractmethod
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu,
    ): ...

    @abstractmethod
    def train(): ...

    @abstractmethod
    def index(self, name: str, collection: list[str]): ...

    @abstractmethod
    def add_to_index(self): ...

    @abstractmethod
    def search(self, name: str, query: Union[str, list[str]]): ...

    @abstractmethod
    def _search(self, name: str, query: str): ...

    @abstractmethod
    def _batch_search(self, name: str, queries: list[str]): ...

    def _invert_pid_docid_map(self) -> Dict[str, List[int]]:
        d = defaultdict(list)
        for k, v in self.pid_docid_map.items():
            d[v].append(k)
        return d

    def _get_collection_files_from_disk(self, index_path: str):
        self.collection = srsly.read_json(index_path / "collection.json")
        if os.path.exists(str(index_path / "docid_metadata_map.json")):
            self.docid_metadata_map = srsly.read_json(
                str(index_path / "docid_metadata_map.json")
            )
        else:
            self.docid_metadata_map = None

        try:
            self.pid_docid_map = srsly.read_json(str(index_path / "pid_docid_map.json"))
        except FileNotFoundError as err:
            raise FileNotFoundError(
                "ERROR: Could not load pid_docid_map from index!",
                "This is likely because you are loading an older, incompatible index.",
            ) from err

        # convert all keys to int when loading from file because saving converts to str
        self.pid_docid_map = {
            int(key): value for key, value in self.pid_docid_map.items()
        }
        self.docid_pid_map = self._invert_pid_docid_map()
