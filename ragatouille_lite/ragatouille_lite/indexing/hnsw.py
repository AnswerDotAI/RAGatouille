import time
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Literal, Optional, TypeVar, Union

import srsly
import torch
from voyager import Index, Space

from ragatouille_lite.index.base import ModelIndex


class HNSWModelIndex(ModelIndex):
    _DEFAULT_INDEX_BSIZE = 32
    index_type = "HNSW"

    def __init__(self, config: ColBERTConfig) -> None:
        super().__init__(config)
        self.config = config
        self.collection = None
        self.pid_docid_map = None
        self.docid_pid_map = None
        self.docid_metadata_map = None
        self.index_root = None

    @staticmethod
    def construct(
        config: ColBERTConfig,
        checkpoint: str,
        collection: List[str],
        index_name: Optional["str"] = None,
        verbose: bool = True,
        **kwargs,
    ) -> "ModelIndex": ...

    @staticmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        index_config: dict[str, Any],
        config: ColBERTConfig,
        verbose: bool = True,
    ) -> "ModelIndex": ...

    def load_from_hf(
        index_name_or_path: str,
    ) -> "ModelIndex": ...

    def push_to_hub(self, repo_id: str, **kwargs) -> None:
        pass

    def export_config(self) -> dict[str, Any]:
        return self.config
