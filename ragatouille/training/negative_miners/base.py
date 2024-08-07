from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class HardNegativeMiner(ABC):
    @abstractmethod
    def export_index(self, path: Union[str, Path]) -> bool:
        ...

    @abstractmethod
    def mine_hard_negatives(
        self,
        queries: list[str],
        collection: list[str],
        neg_k: int,
    ):
        ...

    @abstractmethod
    def _mine(
        self,
        queries: list[str],
        k: int,
    ):
        ...
