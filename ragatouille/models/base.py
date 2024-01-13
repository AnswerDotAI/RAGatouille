from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class LateInteractionModel(ABC):
    @abstractmethod
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu,
    ):
        ...

    @abstractmethod
    def train():
        ...

    @abstractmethod
    def index(self, name: str, collection: list[str]):
        ...

    @abstractmethod
    def add_to_index(self):
        ...

    @abstractmethod
    def search(self, name: str, query: Union[str, list[str]]):
        ...

    @abstractmethod
    def _search(self, name: str, query: str):
        ...

    @abstractmethod
    def _batch_search(self, name: str, queries: list[str]):
        ...
