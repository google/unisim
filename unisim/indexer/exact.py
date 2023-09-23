from typing import Sequence

from .indexer import Indexer
from ..dataclass import Result
from .. import backend as B

# FIXME: use TF or usearch based on accelerator/backend


class ExactIndexer(Indexer):

    def __init__(self) -> None:
        super().__init__()

    def index(self, embebdding) -> bool:
        raise NotImplementedError

    def batch_index(self, embeddings) -> bool:
        raise NotImplementedError

    def query(self, embedding) -> Sequence[Result]:
        raise NotImplementedError

    def batch_query(self, embeddings) -> Sequence[Sequence[Result]]:
        raise NotImplementedError

    def reset(self) -> bool:
        raise NotImplementedError

    def save(self, path: str) -> bool:
        raise NotImplementedError

    def load(self, path: str) -> bool:
        raise NotImplementedError
