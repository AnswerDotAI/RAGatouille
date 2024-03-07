import os
import random
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Literal, Union

import srsly


class TrainingDataProcessor:
    def __init__(
        self,
        collection: list[str],
        queries: list[str],
        negative_miner=None,
    ):
        self.collection = collection
        self.queries = queries
        self.negative_miner = negative_miner
        self._make_data_map()
        self.training_triplets = []

    def process_raw_data(
        self,
        raw_data,
        data_type: Literal["pairs", "triplets", "labeled_pairs"],
        data_dir: Union[str, Path],
        export: bool = True,
        mine_hard_negatives: bool = True,
        num_new_negatives: int = 10,
        positive_label: int = 1,
        negative_label: int = 0,
        hard_negative_minimum_rank: int = 10,
    ):
        if self.negative_miner is None and mine_hard_negatives:
            raise ValueError(
                "mine_hard_negatives is True but no negative miner was provided!"
            )
        if self.negative_miner:
            self.negative_miner.min_rank = hard_negative_minimum_rank
        if data_type == "pairs":
            self._process_raw_pairs(
                raw_data=raw_data,
                mine_hard_negatives=mine_hard_negatives,
                n_new_negatives=num_new_negatives,
            )
        elif data_type == "labeled_pairs":
            self._process_raw_labeled_pairs(
                raw_data=raw_data,
                mine_hard_negatives=mine_hard_negatives,
                n_new_negatives=num_new_negatives,
                positive_label=positive_label,
                negative_label=negative_label,
            )
        elif data_type == "triplets":
            self._process_raw_triplets(
                raw_data=raw_data,
                mine_hard_negatives=mine_hard_negatives,
                n_new_negatives=num_new_negatives,
            )

        if export:
            self.export_training_data(data_dir)

    def _make_individual_triplets(self, query, positives, negatives):
        """Create the training data in ColBERT(v1) format from raw lists of triplets"""
        if len(positives) == 0 or len(negatives) == 0:
            return []
        triplets = []
        q = self.query_map[query]
        random.seed(42)
        if len(positives) > 1:
            all_pos_texts = [p for p in positives]
            max_triplets_per_query = 20
            negs_per_positive = max(1, max_triplets_per_query // len(all_pos_texts))
            initial_triplets_count = 0
            for pos in all_pos_texts:
                p = self.passage_map[pos]
                chosen_negs = random.sample(
                    negatives, min(len(negatives), negs_per_positive)
                )
                for neg in chosen_negs:
                    n = self.passage_map[neg]
                    initial_triplets_count += 1
                    triplets.append([q, p, n])

            extra_triplets_needed = max_triplets_per_query - initial_triplets_count
            if extra_triplets_needed > 0:
                all_combinations = list(product(all_pos_texts, negatives))
                random.seed(42)
                random.shuffle(all_combinations)
                for pos, neg in all_combinations:
                    p = self.passage_map[pos]
                    n = self.passage_map[neg]
                    if [q, p, n] not in triplets:
                        triplets.append([q, p, n])
                        extra_triplets_needed -= 1
                        if extra_triplets_needed <= 0:
                            break

        else:
            p = self.passage_map[positives[0]]
            for n in negatives:
                triplets.append([q, p, self.passage_map[n]])

        return triplets

    def _get_new_negatives(self, query, passages, mine_hard_negatives, n_new_negatives):
        """Generate new negatives for each query, using either:
        - The assigned hard negative miner if mine_hard_negatives is True
        - Randomly sampling from the full collection otherwise
        """
        if mine_hard_negatives:
            hard_negatives = self.negative_miner.mine_hard_negatives(query)
            candidates = [
                x
                for x in hard_negatives
                if x not in passages["positives"] and x not in passages["negatives"]
            ]
            new_negatives = random.sample(
                candidates,
                min(n_new_negatives, len(candidates)),
            )
        else:
            new_negatives = [
                x
                for x in random.sample(
                    self.collection, min(n_new_negatives, len(self.collection))
                )
                if x not in passages["positives"] and x not in passages["negatives"]
            ]

        return new_negatives

    def _process_raw_pairs(self, raw_data, mine_hard_negatives, n_new_negatives):
        """Convert unlabeled pairs into training triplets.
        It's assumed unlabeled pairs are always in the format (query, relevant_passage)"""
        training_triplets = []
        raw_grouped_triplets = defaultdict(lambda: defaultdict(list))

        for query, positive in raw_data:
            if isinstance(positive, str):
                positive = [positive]
            elif isinstance(positive, dict):
                positive = [positive["content"]]
            raw_grouped_triplets[query]["positives"] += positive

        for query, passages in raw_grouped_triplets.items():
            if n_new_negatives > 0:
                passages["negatives"] += self._get_new_negatives(
                    query=query,
                    passages=passages,
                    mine_hard_negatives=mine_hard_negatives,
                    n_new_negatives=n_new_negatives,
                )
            training_triplets += self._make_individual_triplets(
                query=query,
                positives=list(set(passages["positives"])),
                negatives=list(set(passages["negatives"])),
            )
        self.training_triplets = training_triplets

    def _process_raw_labeled_pairs(
        self,
        raw_data,
        mine_hard_negatives,
        n_new_negatives,
        positive_label,
        negative_label,
    ):
        """
        Convert labeled pairs intro training triplets.
        Labeled pairs are in the format (query, passage, label)
        """
        training_triplets = []
        raw_grouped_triplets = defaultdict(lambda: defaultdict(list))

        for query, passage, label in raw_data:
            if isinstance(passage, str):
                passage = [passage]
            if label == positive_label:
                label = "positives"
            elif label == negative_label:
                label = "negatives"
            else:
                raise ValueError(
                    f"Label {label} must correspond to either positive_label or negative_label!"
                )

            raw_grouped_triplets[query][label] += passage

        for query, passages in raw_grouped_triplets.items():
            if n_new_negatives > 0:
                passages["negatives"] += self._get_new_negatives(
                    query=query,
                    passages=passages,
                    mine_hard_negatives=mine_hard_negatives,
                    n_new_negatives=n_new_negatives,
                )

            training_triplets += self._make_individual_triplets(
                query=query,
                positives=passages["positives"],
                negatives=passages["negatives"],
            )
        self.training_triplets = training_triplets

    def _process_raw_triplets(self, raw_data, mine_hard_negatives, n_new_negatives):
        """
        Convert raw triplets
        (query, positives : str | list[str], negatives: str | list[str])
        into training triplets.
        """
        training_triplets = []
        raw_grouped_triplets = defaultdict(lambda: defaultdict(list))
        for query, positive, negative in raw_data:
            if isinstance(positive, str):
                positive = [positive]
            if isinstance(negative, str):
                negative = [negative]

            raw_grouped_triplets[query]["positives"] += positive
            raw_grouped_triplets[query]["negatives"] += negative

        for query, passages in raw_grouped_triplets.items():
            if n_new_negatives > 0:
                passages["negatives"] += self._get_new_negatives(
                    query=query,
                    passages=passages,
                    mine_hard_negatives=mine_hard_negatives,
                    n_new_negatives=n_new_negatives,
                )
            training_triplets += self._make_individual_triplets(
                query=query,
                positives=passages["positives"],
                negatives=passages["negatives"],
            )
        self.training_triplets = training_triplets

    def _make_data_map(self):
        """
        Generate a query_text: query_id and passage_text: passage_id mapping
        To easily generate ColBERT-format training data.
        """
        self.query_map = {}
        self.passage_map = {}

        for i, query in enumerate(self.queries):
            self.query_map[query] = i
        for i, passage in enumerate(list(self.collection)):
            self.passage_map[passage] = i

    def export_training_data(self, path: Union[str, Path]):
        """
        Export training data for both training and versioning purposes.
        {path} should ideally be dvc versioned.
        """

        path = Path(path)

        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)

        with open(path / "queries.train.colbert.tsv", "w") as f:
            for query, idx in self.query_map.items():
                query = query.replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{query}\n")
        with open(path / "corpus.train.colbert.tsv", "w") as f:
            for document, idx in self.passage_map.items():
                document = document.replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{document}\n")

        random.seed(42)
        random.shuffle(self.training_triplets)
        srsly.write_jsonl(path / "triples.train.colbert.jsonl", self.training_triplets)
