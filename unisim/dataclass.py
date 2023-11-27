# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import dataclasses as dc
from typing import Any, List

from .types import Embedding


@dc.dataclass
class Match:
    """Metadata associated with a query match.

    Attributes:
        idx: Idx of the match, as stored in the index.

        data: Input data of the match, if stored.

        rank: Rank of the match with respect to the query similarity, e.g. 0 for most similar match.

        similarity: Cosine similarity from the match to the query.

        embedding: Embedding of the match data.

        is_match: Whether the match and the query are near-duplicate matches.
    """

    idx: int
    data: Any | None = None
    rank: int | None = None
    similarity: float | None = None
    embedding: Embedding | None = None
    is_match: bool = False


@dc.dataclass
class Result:
    """Metadata associated a search result for a particular query.

    Attributes:
        query_idx: Idx of the query in the list of queries passed to UniSim's search function.

        query_data: Input data of the query.

        query_embedding: Embedding of the query data.

        num_matches: Number of near-duplicate matches for query found.

        matches: A list of Match objects corresponding to the near-duplicate matches found for the query.
    """

    query_idx: int
    query_data: Any | None = None
    query_embedding: Embedding | None = None
    num_matches: int = 0
    matches: List[Match] = dc.field(default_factory=lambda: [])


@dc.dataclass
class ResultCollection:
    """Collection of search results returned.

    Attributes:
        total_matches: Total number of near-duplicate matches found across all search queries.

        results: A list of Result objects corresponding to each search query result.
    """

    total_matches: int = 0
    results: List[Result] = dc.field(default_factory=lambda: [])

    def __repr__(self) -> str:
        res = "-[Results Statistics]-\n"
        res += f"Number of Results: {len(self.results)}\n"
        res += f"Total Matches: {self.total_matches}\n"
        return res

    def merge_result_collection(self, other: ResultCollection) -> ResultCollection:
        self.total_matches += other.total_matches
        self.results.extend(other.results)
        return self
