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
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
from usearch.index import BatchMatches, Index, MetricKind, search

from .dataclass import Match, Result, ResultCollection
from .enums import IndexerType
from .types import BatchEmbeddings


class Indexer:
    def __init__(
        self,
        embedding_size: int,
        similarity_threshold: float,
        index_type: IndexerType,
        params: Dict = {},
    ) -> None:
        self.embedding_size = embedding_size
        self.similarity_threshold = similarity_threshold
        self.index_type = index_type
        self.params = params

        # determine if we use ANN or exact search for the index
        self.use_exact = True if index_type == IndexerType.exact else False

        if self.use_exact:
            # exact matching and we need to store the embeddings in memory
            self.embeddings = []

        else:
            # initializing the USearch ANN index
            self.index = Index(
                ndim=self.embedding_size,
                metric=params.get("metric", "ip"),
                dtype=params.get("dtype", "f32"),
                connectivity=params.get("connectivity", 16),
                expansion_add=params.get("expansion_add", 128),
                expansion_search=params.get("expansion_search", 64),
            )

    def add(self, embeddings: BatchEmbeddings, idxs: List[int]) -> None:
        self.idxs.extend(idxs)

        if self.use_exact:
            self.embeddings.extend(embeddings)
        else:
            keys = np.asanyarray(idxs)
            embs = np.asanyarray(embeddings)
            self.index.add(keys, embs)

    def search(
        self,
        queries: Sequence[Any],
        query_embeddings: BatchEmbeddings,
        k: int,
        return_data: bool,
        data: Sequence[Any] = [],
    ) -> ResultCollection:
        if return_data and not data:
            raise ValueError("Can't return data, data is empty")

        # Using USearch exact search
        if self.use_exact:
            self.embeddings = np.asanyarray(self.embeddings)

            # check how many items to return in case k is larger than index size
            index_size = len(self.embeddings)
            count = k if k < index_size else index_size

            matches_batch = search(
                self.embeddings,
                query_embeddings,
                metric=MetricKind.IP,
                count=count,
                exact=self.use_exact,
            )

        # use USearch ANN search
        else:
            index_size = self.index.size
            count = k if k < index_size else index_size
            matches_batch = self.index.search(query_embeddings, count=count)

        # compute matches
        results_map: Dict[str, Result] = {}
        matches_map: Dict[str, Dict[str, Match]] = defaultdict(dict)

        # for single batch lookup Matches is returned instead of BatchMatches
        if not isinstance(matches_batch, BatchMatches):
            matches_batch = [matches_batch]

        for query_idx, matches in enumerate(matches_batch):
            result = Result(query_idx=query_idx)
            if return_data:
                result.query = queries[query_idx]

            for rank, m in enumerate(matches):
                target_idx = m.key
                similarity = 1 - m.distance

                match = Match(idx=target_idx, global_rank=rank, similarity=similarity)

                if return_data:
                    match.data = data[target_idx]

                # is a match?
                if similarity >= self.similarity_threshold:
                    match.is_match = True
                    result.num_matches += 1

                # save the match in a map to find it back in partial analysis
                matches_map[query_idx][match.idx] = match

            # save in a map to find it back during partial analysis
            results_map[result.query_idx] = result

        # create and return results collection
        results_collection = ResultCollection()
        for result in results_map.values():
            matches = [m for m in matches_map[result.query_idx].values()]
            result.matches = matches
            results_collection.total_matches += result.num_matches
            results_collection.results.append(result)
        return results_collection

    def reset(self) -> bool:
        if self.use_exact:
            self.embeddings = []
        else:
            self.index.reset()

        self.idxs = []
