from typing import Union, Optional
from pathlib import Path
from colbert.infra import Run, ColBERTConfig, RunConfig
from colbert import Indexer, Searcher, Trainer, IndexUpdater
import torch
import srsly

from ragatouille.models.base import LateInteractionModel


class ColBERT(LateInteractionModel):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        index_name: Optional[str] = None,
        verbose: int = 1,
        load_from_index: bool = False,
        **kwargs,
    ):
        self.verbose = verbose
        self.collection = None
        if n_gpu == -1:
            n_gpu = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()

        if load_from_index:
            ckpt_config = ColBERTConfig.load_from_index(
                str(pretrained_model_name_or_path)
            )
            self.config = ckpt_config
            self.run_config = RunConfig(
                nranks=n_gpu, experiment=self.config.experiment, root=self.config.root
            )
            self.checkpoint = self.config.checkpoint
            self.index_name = self.config.index_name
            self.collection = self._get_collection_from_file(
                str(pretrained_model_name_or_path / "collection.json")
            )
        else:
            ckpt_config = ColBERTConfig.load_from_checkpoint(
                str(pretrained_model_name_or_path)
            )
            self.run_config = RunConfig(
                nranks=n_gpu, experiment="colbert", root=".ragatouille/"
            )
            local_config = ColBERTConfig(**kwargs)
            self.config = ColBERTConfig.from_existing(
                ckpt_config,
                local_config,
            )
            self.checkpoint = pretrained_model_name_or_path
            self.index_name = index_name

        self.run_context = Run().context(self.run_config)
        self.run_context.__enter__()  # Manually enter the context
        self.searcher = None

    def _update_index(self, new_documents: list[str], searcher: Searcher):
        updater = IndexUpdater(
            config=self.config, searcher=searcher, checkpoint=self.checkpoint
        )
        updater.add(new_documents)
        updater.persist_to_disk()

    def _get_collection_from_file(self, collection_path: str):
        return srsly.read_json(collection_path)

    def _write_collection_to_file(self, collection, collection_path: str):
        srsly.write_json(collection_path, collection)

    def add_to_index(
        self,
        new_documents: list[str],
        index_name: Optional[str] = None,
    ):
        self.index_name = index_name if index_name is not None else self.index_name
        if self.index_name is None:
            print(
                "Cannot add to index without an index_name! Please provide one.",
                "Returning empty results.",
            )
            return None

        print(
            "WARNING: add_to_index support is currently experimental!",
            "add_to_index support will be more thorough in future versions",
        )

        searcher = Searcher(
            checkpoint=self.checkpoint,
            config=None,
            collection=self.collection,
            index=self.index_name,
            verbose=self.verbose,
        )
        new_documents = list(set(new_documents))
        current_len = len(searcher.collection)
        new_doc_len = len(new_documents)

        if (
            current_len + new_doc_len < 5000
            or new_doc_len > current_len * 0.05
            or current_len + new_doc_len
            > 100  # Export bug handler -- TODO: Remove this requirement
        ):
            new_documents += [x for x in searcher.collection]
            self.index(
                new_documents,
                index_name=self.index_name,
                max_document_length=self.config.doc_maxlen,
                overwrite="force_silent_overwrite",
            )
        else:
            self._update_index(new_documents, searcher)

        print(
            f"Successfully updated index with {new_doc_len} new documents!\n",
            f"New index size: {new_doc_len + current_len}",
        )

        return str(
            Path(self.run_config.root)
            / Path(self.run_config.experiment)
            / "indexes"
            / self.index_name
        )

    def index(
        self,
        collection: list[str],
        index_name: Optional["str"] = None,
        max_document_length: int = 256,
        overwrite: Union[bool, str] = "reuse",
    ):
        self.config.doc_maxlen = max_document_length
        if index_name is not None:
            if self.index_name is not None:
                print(
                    f"New index_name received!",
                    f"Updating current index_name ({self.index_name}) to {index_name}",
                )
            self.index_name = index_name
        else:
            if self.index_name is None:
                print(
                    f"No index_name received!",
                    f"Using default index_name ({self.checkpoint}_new_index)",
                )
            self.index_name = self.checkpoint + "new_index"

        collection = list(set(collection))
        self.collection = collection

        nbits = 2
        if len(collection) < 5000:
            nbits = 8
        elif len(collection) < 10000:
            nbits = 4
        self.config = ColBERTConfig.from_existing(
            self.config, ColBERTConfig(nbits=nbits)
        )
        self.indexer = Indexer(
            checkpoint=self.checkpoint,
            config=self.config,
            verbose=self.verbose,
        )
        self.indexer.index(
            name=self.index_name, collection=collection, overwrite=overwrite
        )

        index_path = str(
            Path(self.run_config.root)
            / Path(self.run_config.experiment)
            / "indexes"
            / self.index_name
        )
        self._write_collection_to_file(collection, index_path + "/collection.json")
        print("Done indexing!")

    def _load_searcher(
        self,
        index_name: Optional[str],
        force_fast: bool = False,
    ):
        if index_name is not None:
            if self.index_name is not None:
                print(
                    f"New index_name received!",
                    f"Updating current index_name ({self.index_name}) to {index_name}",
                )
            self.index_name = self.index_name
        else:
            if self.index_name is None:
                print(
                    "Cannot search without an index_name! Please provide one.",
                    "Returning empty results.",
                )
                return None
        print(
            f"Loading searcher for index {self.index_name} for the first time...",
            "This may take a few seconds",
        )
        self.searcher = Searcher(
            checkpoint=self.checkpoint,
            config=None,
            collection=self.collection,
            index=self.index_name,
        )

        if not force_fast:
            if len(self.searcher.collection) < 10000:
                self.searcher.configure(ncells=4)
                self.searcher.configure(centroid_score_threshold=0.4)
                self.searcher.configure(ndocs=512)
            elif len(self.searcher.collection) < 100000:
                self.searcher.configure(ncells=2)
                self.searcher.configure(centroid_score_threshold=0.45)
                self.searcher.configure(ndocs=1024)
            # Otherwise, use defaults for k
        else:
            # Use fast settingss
            self.searcher.configure(ncells=1)
            self.searcher.configure(centroid_score_threshold=0.5)
            self.searcher.configure(ndocs=256)

        print("Searcher loaded!")

    def search(
        self,
        query: Union[str, list[str]],
        index_name: Optional["str"] = None,
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
    ):
        if self.searcher is None or (
            index_name is not None and self.index_name != index_name
        ):
            self._load_searcher(index_name=index_name, force_fast=force_fast)

        if isinstance(query, str):
            results = [self._search(query, k)]
        else:
            results = self._batch_search(query, k)

        to_return = []

        for result in results:
            result_for_query = []
            for id_, rank, score in zip(*result):
                result_for_query.append(
                    {
                        "content": self.searcher.collection[id_],
                        "score": score,
                        "rank": rank - 1 if zero_index_ranks else rank,
                    }
                )
            to_return.append(result_for_query)

        if len(to_return) == 1:
            return to_return[0]
        return to_return

    def _search(self, query: str, k: int):
        return self.searcher.search(query, k=k)

    def _batch_search(self, query: list[str], k: int):
        queries = {i: x for i, x in enumerate(query)}
        results = self.searcher.search_all(queries, k=k)
        results = [
            [list(zip(*value))[i] for i in range(3)]
            for value in results.todict().values()
        ]
        return results

    def train(self, data_dir, training_config: ColBERTConfig):
        training_config = ColBERTConfig.from_existing(self.config, training_config)
        training_config.nway = 2
        with Run().context(self.run_config):
            trainer = Trainer(
                triples=str(data_dir / "triples.train.colbert.jsonl"),
                queries=str(data_dir / "queries.train.colbert.tsv"),
                collection=str(data_dir / "corpus.train.colbert.tsv"),
                config=training_config,
            )

            trainer.train(checkpoint=self.checkpoint)

    def __del__(self):
        # Clean up context
        self.run_context.__exit__(None, None, None)
