from abc import ABC, abstractmethod
from typing import Sequence

from ..dataclass import Result


class Indexer(ABC):

    @abstractmethod
    def index(self, embebdding) -> bool:
        pass

    @abstractmethod
    def batch_index(self, embeddings) -> bool:
        pass

    @abstractmethod
    def query(self, embedding) -> Sequence[Result]:
        pass

    @abstractmethod
    def batch_query(self, embeddings) -> Sequence[Sequence[Result]]:
        pass

    @abstractmethod
    def reset(self) -> bool:
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        pass
