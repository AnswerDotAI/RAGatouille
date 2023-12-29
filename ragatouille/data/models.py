from pydantic import BaseModel


class TrainingTriplet(BaseModel):
    """
    A training triplet.
    """

    anchor: str
    positive: str
    negative: str


class QueryPassages(BaseModel):
    """
    A query and a list of passages.
    """

    query: str
    positive_passages: list[str]
    negative_passages: list[str]
