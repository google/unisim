from __future__ import annotations

import dataclasses as dc
import numpy as np

from typing import Optional, Dict, Any, Callable, Sequence, List
from .types import GlobalEmbedding, PartialEmbeddings, TensorEmbedding


EqFunc = Callable[[Any, Any], bool]


def _optional_eq(a: Any, b: Any, eq_fun: EqFunc) -> bool:
    """__eq__ for Optional[Any] types."""
    if a is None:
        return b is None
    elif b is None:
        return False

    return eq_fun(a, b)


def _basic_eq(a: Any, b: Any) -> bool:
    eq: bool = a == b
    return eq


def _ndarray_eq(a: TensorEmbedding, b: TensorEmbedding) -> bool:
    eq: bool = np.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    return eq


@dc.dataclass
class Embedding:
    rank: int  # NN position
    distance: float
    embedding: TensorEmbedding
    content_index: int  # where the content start
    content_length: int  # how long is the content


@dc.dataclass
class PartialMatch:
    distance: float | None   # closest distance
    match_len: int | None  # len of the chunks

    target_idx: int | None  # closest
    target_chunk_idx: int | None
    target_match_position: int | None

    query_match_position: int | None

    is_partial_match: bool = False


@dc.dataclass
class Similarity:
    query_embedding: TensorEmbedding
    target_embedding: TensorEmbedding
    distance: float

    is_match: bool = False  # is near duplicate or partial match or both
    is_near_duplicate: bool = False
    is_partial_match: bool = False

    num_partial_matches: int = 0
    partial_matches_ratio: float = 0  # num matched partial / total chunk
    partial_matches: List[PartialMatch] = dc.field(default_factory=lambda: [])


@dc.dataclass
class Result:
    """Result returned by Unisim.
    # FIXME when stabilized
    Attributes:

        match: the result is matching the query either as near-duplicate or
        having significant content in common (partial matchs)

        near_duplicate: true if the match is a near duplicate: that is
        if the result distance is above the similarity threshold for the
        global embedding.

        partial_match:

        partial_match_ratio:

        partial_match_distance:

        rank: Rank of the match with respect to the global distance.

        distance: The distance from the match to the input.

        label: The label associated with the match. Default None.

        embedding: The embedded match vector

        data: the metadata associated with the match result.
        Default None.
    """
    is_match: bool | None  # is near duplicate or partial match or both
    is_near_duplicate: bool | None
    is_partial_match: bool | None
    partial_match_ratio: float | None  # num matched partial / total chunk

    global_embedding: Embedding | None
    partial_embeddings:  Sequence[Embedding]

    # optional data
    idx: Optional[int] = dc.field(default=None)
    data: Optional[Dict] = dc.field(default=None)

    # FIXME __repr__

    # FIXME: __eq__
    # def __eq__(self, other) -> bool:
    #     if other.__class__ is not self.__class__:
    #         return False
    #     if self.rank != other.rank:
    #         return False
    #     elif self.distance != other.distance:
    #         return False
    #     elif not _optional_eq(self.label, other.label, _basic_eq):
    #         return False
    #     elif not _optional_eq(self.embedding, other.embedding, _ndarray_eq):
    #         return False
    #     elif not _optional_eq(self.data, other.data, _basic_eq):
    #         return False
    #     else:
    #         return True
