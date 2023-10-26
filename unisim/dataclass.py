"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .types import GlobalEmbedding, PartialEmbeddings, TensorEmbedding


@dc.dataclass
class Match:
    idx: int
    global_rank: int | None = None
    global_similarity: float | None = None

    # todo we need to count num partials
    # num chunk
    # give partial match ratio
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


@dc.dataclass
class ResultCollection:
    total_global_matches: int = 0
    total_partial_matches: int = 0
    results: List[Result] = dc.field(default_factory=lambda: [])

    def __repr__(self) -> str:
        res = "-[Results statistics]-"
        res = f"Results: {len(self.results)}\n"
        res += f"Global Match: {self.total_global_matches}\n"
        res += f"Partial Match: {self.total_partial_matches}"
        return res

    def merge_result_collection(self, other: ResultCollection) -> ResultCollection:
        self.total_global_matches += other.total_global_matches
        self.total_partial_matches += other.total_partial_matches
        self.results.extend(other.results)
        return self


@dc.dataclass
class Similarity:
    query_embedding: TensorEmbedding
    target_embedding: TensorEmbedding
    # TODO(ovallis) this is the distance of the similarity?
    distance: float

    is_global_match: bool = False
    is_partial_match: bool = False

    num_partial_matches: int = 0
    partial_matches_ratio: float = 0  # num matched partial / total chunk
    partial_matches: List[Match] = dc.field(default_factory=lambda: [])
