from pathlib import Path
from typing import Union, Literal, Optional

from colbert.infra import ColBERTConfig

from ragatouille.models import LateInteractionModel, ColBERT
from ragatouille.negative_miners import HardNegativeMiner, SimpleMiner
from ragatouille.utils import seeded_shuffle
from ragatouille.data import TrainingDataProcessor


class RAGTrainer:
    """Main trainer to fine-tune/train ColBERT models with a few lines."""

    model: Union[LateInteractionModel, None] = None
    negative_miner: Union[HardNegativeMiner, None] = None
    collection: list[str] = []
    queries: Union[list[str], None] = None
    raw_data: Union[list[tuple], list[list], None] = None
    training_triplets: list[list[int]] = list()

    def __init__(
        self,
        model_name: str,
        pretrained_model_name: str,
        language_code: str = "en",
        n_usable_gpus: int = -1,
    ):
        """
        Initialise a RAGTrainer instance. This will load a base model: either an existing ColBERT model to fine-tune or a BERT/RoBERTa-like model to build a new ColBERT model from.

        Parameters:
            model_name: str - Name of the model to train. This will be used to name the checkpoints and the index.
            pretrained_model_name: str - Name of the pretrained model to use as a base. Can be a local path to a checkpoint or a huggingface model name.
            language_code: str - Language code of the model to train. This will be used to name the checkpoints and the index.
            n_usable_gpus: int - Number of GPUs to use. By default, value is -1, which means use all available GPUs or none if no GPU is available.

        Returns:
            self (RAGTrainer): The current instance of RAGTrainer, with the base model initialised.
        """

        self.model_name = model_name
        self.pretrained_model_name = pretrained_model_name
        self.language_code = language_code
        self.model = ColBERT(
            pretrained_model_name_or_path=pretrained_model_name, n_gpu=n_usable_gpus
        )

    def add_documents(self, documents: list[str]):
        self.collection += documents
        seeded_shuffle(self.collection)

    def export_training_data(self, path: Union[str, Path]):
        """
        Manually export the training data processed by prepare_training_data to a given path.

        Parameters:
            path: Union[str, Path] - Path to the directory where the data will be exported."""
        self.data_processor.export_training_data(path)

    def prepare_training_data(
        self,
        raw_data: Union[list[tuple], list[list]],
        all_documents: Optional[list[str]] = None,
        data_out_path: Union[str, Path] = "./data/",
        num_new_negatives: int = 10,
        hard_negative_minimum_rank: int = 10,
        mine_hard_negatives: bool = True,
        hard_negative_model_size: str = "small",
        pairs_with_labels: bool = False,
        positive_label: Union[int, str] = 1,
        negative_label: Union[int, str] = 0,
    ) -> str:
        """
        Fully pre-process input-data in various raw formats into ColBERT-ready files and triplets.
        Will accept a variety of formats, such as unannotated pairs, annotated pairs, triplets of strings and triplets of list of strings.
        Will process into a ColBERT-ready format and export to data_out_path.
        Will generate hard negatives if mine_hard_negatives is True.
        num_new_negatives decides how many negatives will be generated. if mine_hard_negatives is False and num_new_negatives is > 0, these negatives will be randomly sampled.

        Parameters:
            raw_data: Union[list[tuple], list[list]] - List of pairs, annotated pairs, or triplets of strings.
            all_documents: Optional[list[str]] - A corpus of documents to be used for sampling negatives.
            data_out_path: Union[str, Path] - Path to the directory where the data will be exported (can be a tmp directory).
            num_new_negatives: int - Number of new negatives to generate for each query.
            mine_hard_negatives: bool - Whether to use hard negatives mining or not.
            hard_negative_model_size: str - Size of the model to use for hard negatives mining.
            pairs_with_labels: bool - Whether the raw_data is a list of pairs with labels or not.
            positive_label: Union[int, str] - Label to use for positive pairs.
            negative_label: Union[int, str] - Label to use for negative pairs.

        Returns:
            data_out_path: Union[str, Path] - Path to the directory where the data has been exported.
        """
        if all_documents is not None:
            self.collection += all_documents

        self.data_dir = Path(data_out_path)
        if len(raw_data[0]) == 2:
            data_type = "pairs"
            if pairs_with_labels:
                data_type = "labeled_pairs"
        elif len(raw_data[0]) == 3:
            data_type = "triplets"
        else:
            raise ValueError("Raw data must be a list of pairs or triplets of strings.")
        self.collection += [x[1] for x in raw_data]
        if data_type == "triplets":
            self.collection += [x[2] for x in raw_data]

        self.queries = set([x[0] for x in raw_data])
        self.collection = list(set(self.collection))
        seeded_shuffle(self.collection)

        if mine_hard_negatives:
            self.negative_miner = SimpleMiner(
                language_code=self.language_code,
                model_size=hard_negative_model_size,
            )
            self.negative_miner.build_index(self.collection)

        self.data_processor = TrainingDataProcessor(
            collection=self.collection,
            queries=self.queries,
            negative_miner=self.negative_miner if mine_hard_negatives else None,
        )

        self.data_processor.process_raw_data(
            data_type=data_type,
            raw_data=raw_data,
            export=True,
            data_dir=data_out_path,
            num_new_negatives=num_new_negatives,
            positive_label=positive_label,
            negative_label=negative_label,
            mine_hard_negatives=mine_hard_negatives,
            hard_negative_minimum_rank=hard_negative_minimum_rank,
        )

        self.training_triplets = self.data_processor.training_triplets

        return data_out_path

    def train(
        self,
        batch_size: int = 32,
        nbits: int = 2,
        maxsteps: int = 500_000,
        use_ib_negatives: bool = True,
        learning_rate: float = 5e-6,
        dim: int = 128,
        doc_maxlen: int = 256,
        use_relu: bool = False,
        warmup_steps: Union[int, Literal["auto"]] = "auto",
        accumsteps: int = 1,
    ) -> str:
        """
        Launch training or fine-tuning of a ColBERT model.
        Parameters:
            batch_size: int - Total batch size -- divice by n_usable_gpus for per-GPU batch size.
            nbits: int - number of bits used for vector compression by the traiened model. 2 is usually ideal.
            maxsteps: int - End training early afte maxsteps steps.
            use_ib_negatives: bool - Whether to use in-batch negatives to calculate loss or not.
            learning_rate: float - ColBERT litterature usually has this performing best between 3e-6 - 2e-5 depending on data size
            dim: int - Size of individual vector representations.
            doc_maxlen: int - The maximum length after which passages will be truncated
            warmup_steps: Union[int, Literal["auto"]] - How many warmup steps to use for the learning rate.
                                                      Auto will default to 10% of total steps
            accumsteps: How many gradient accummulation steps to use to simulate higher batch sizes.

        Returns:
            model_path: str - Path to the trained model.
        """
        if not self.training_triplets:
            total_triplets = sum(
                1 for _ in open(str(self.data_dir / "triples.train.colbert.jsonl"), "r")
            )
        else:
            total_triplets = len(self.training_triplets)

        training_config = ColBERTConfig(
            bsize=batch_size,
            model_name=self.model_name,
            name=self.model_name,
            checkpoint=self.pretrained_model_name,
            use_ib_negatives=use_ib_negatives,
            maxsteps=maxsteps,
            nbits=nbits,
            lr=learning_rate,
            dim=dim,
            doc_maxlen=doc_maxlen,
            relu=use_relu,
            accumsteps=accumsteps,
            warmup=int(total_triplets // batch_size * 0.1)
            if warmup_steps == "auto"
            else warmup_steps,
            save_every=int(total_triplets // batch_size // 10),
        )

        return self.model.train(data_dir=self.data_dir, training_config=training_config)
