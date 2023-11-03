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
from typing import Any, List


@dc.dataclass
class Match:
    idx: int
    rank: int | None = None
    similarity: float | None = None
    is_match: bool = False

    # store data
    data: Any = dc.field(default=None)

    # TODO (marinazh): allow metadata?

    # TODO (marinazh): add in partial match positions
    # partial_target_match_position: int = 0
    # partial_query_matches_position: int = 0
    # partial_matches_ratio: int = 0


@dc.dataclass
class Result:
    query_idx: int
    query: str = ""
    num_matches: int = 0
    matches: List[Match] = dc.field(default_factory=lambda: [])


@dc.dataclass
class ResultCollection:
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
