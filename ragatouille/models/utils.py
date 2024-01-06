from pathlib import Path
import random
from transformers import BertPreTrainedModel, AutoModel
from colbert.infra import ColBERTConfig
import torch
import torch.nn as nn
from colbert.modeling.colbert import ColBERT


def seeded_shuffle(collection: list, seed: int = 42):
    random.seed(seed)
    random.shuffle(collection)
    return collection


""" HUGGINGFACE """


def export_to_huggingface_hub(
    colbert_path: str | Path,
    huggingface_repo_name: str,
    export_vespa_onnx: bool = False,
):
    colbert_config = ColBERTConfig.load_from_checkpoint(colbert_path)
    assert colbert_config is not None
    colbert_model = ColBERT(
        colbert_path,
        colbert_config=colbert_config,
    )
    try:
        save_model = colbert_model.save
    except Exception:
        save_model = colbert_model.module.save
    save_model(".tmp/hugging_face_export")
    # TODO
    if export_vespa_onnx:
        export_to_vespa_onnx(colbert_path, out_path=".tmp/hugging_face_export")


""" VESPA """


class VespaColBERT(BertPreTrainedModel):
    def __init__(self, config, dim):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)


def export_to_vespa_onnx(
    colbert_path: str | Path,
    out_path: str | Path,
):
    out_path = Path(out_path)
    vespa_colbert = VespaColBERT.from_pretrained(colbert_path, dim=128)
    input_names = ["input_ids", "attention_mask"]
    output_names = ["contextual"]
    input_ids = torch.ones(1, 32, dtype=torch.int64)
    attention_mask = torch.ones(1, 32, dtype=torch.int64)
    args = (input_ids, attention_mask)
    torch.onnx.export(
        vespa_colbert,
        args=args,
        f=str(out_path / "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "batch"},
            "attention_mask": {0: "batch", 1: "batch"},
            "contextual": {0: "batch", 1: "batch"},
        },
        opset_version=17,
    )
