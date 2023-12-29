from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class HardNegativeMiner(ABC):
    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def build_index(
        self, collection: list[str], batch_size: int, save_index: bool, path: str | Path
    ) -> Any:
        ...

    @abstractmethod
    def export_index(self, path: str | Path) -> bool:
        ...
