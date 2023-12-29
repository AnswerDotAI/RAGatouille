from .models import ColBERT


class RAGTrainer:
    def __init__(self):
        self.model = None
        self.negative_miner = None
        self.index = None
        self.collection = None
        self.queries = None
        self.training = None
