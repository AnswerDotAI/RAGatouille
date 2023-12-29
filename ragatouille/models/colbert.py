from typing import Union
from pathlib import Path
from colbert import Run, ColBERTConfig, Indexer, RunConfig
import torch


class ColBERT:
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        **kwargs,
    ):
        if n_gpu == -1:
            n_gpu = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()
        run_config = RunConfig(
            nranks=n_gpu, experiment="colbert", root="/.ragatouille/"
        )
        self.run_context = Run().context(run_config)
        self.run_context.__enter__()  # Manually enter the context
        self.checkpoint = pretrained_model_name_or_path
        ckpt_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        local_config = ColBERTConfig(**kwargs)
        self.config = ColBERTConfig.from_existing(
            ckpt_config,
            local_config,
        )

    def train():
        pass

    def index(self, name, collection):
        pass
        self.indexer = Indexer(checkpoint="/path/to/checkpoint", config=self.config)
        self.indexer.index(name=name, collection=collection)

    def search(self, name, query):
        pass

    def __del__(self):
        # Clean up the context if needed
        self.run_context.__exit__(None, None, None)
