from __future__ import annotations

import dataclasses as dc
import numpy as np

from typing import Optional, Dict, Any, Callable, Sequence, List
from .types import GlobalEmbedding, PartialEmbeddings, TensorEmbedding


@dc.dataclass
class Match:
    idx: int
    global_rank: int | None = None
    global_similarity: float | None = None

    # todo move to partial map @ovallis
    partial_rank: int | None = None
    partial_similarity: float | None = None

    is_global_match: bool = False
    is_partial_match: bool = False

    # partial match only
    match_len: int = 0
    target_match_position: int = 0
    query_match_position: int = 0

    # if we have the content
    data: Any = dc.field(default=None)


@dc.dataclass
class Result:
    query_idx: int
    query: str = ""
    num_global_matches: int = 0
    num_partial_matches: int = 0
    matches: List[Match] = dc.field(default_factory=lambda: [])

# fixme Result collection with export to pandas


@dc.dataclass
class Similarity:
    query_embedding: TensorEmbedding
    target_embedding: TensorEmbedding
    distance: float

    is_global_match: bool = False
    is_partial_match: bool = False

    num_partial_matches: int = 0
    partial_matches_ratio: float = 0  # num matched partial / total chunk
    partial_matches: List[Match] = dc.field(default_factory=lambda: [])
