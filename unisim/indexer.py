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
from typing import Any, Dict, Sequence

import numpy as np
from usearch.index import Index
from usearch.index import MetricKind, search

from .dataclass import Match, Result, ResultCollection
from .enums import IndexerType


class Indexer:

    def __init__(
        self,
        embedding_size: int,
        global_threshold: float,
        partial_threshold: float,
        index_type: IndexerType,
        params: Dict,
    ) -> None:
        self.embedding_size = embedding_size
        self.global_threshold = global_threshold
        self.partial_threshold = partial_threshold
        self.index_type = index_type
        self.params = params

        # multiple embeddings (partial_embeddings) can be long to same idxs
        self.global_idxs = []
        self.partial_idx_2_global_idx = []

        # each partial needs a unique keys so we track that
        self.usearch_pkeys_count = 0

        # determine what type of index we want
        self.use_exact = True if index_type == IndexerType.exact else False

        if self.use_exact:
            # exact matching and we need to store the embeddings in memory
            self.global_embeddings = []
            self.partial_embeddings = []

        else:
            # initializing the usearch ANN indexers
            self.global_index = Index(
                ndim=self.embedding_size,
                metric=params.get("metric", "ip"),
                dtype=params.get("dtype", "f32"),
                connectivity=params.get("connectivity", 16),
                expansion_add=params.get("expansion_add", 128),
                expansion_search=params.get("expansion_search", 64),
            )
            self.partial_index = Index(
                ndim=self.embedding_size,
                metric=params.get("metric", "ip"),
                dtype=params.get("dtype", "f32"),
                connectivity=params.get("connectivity", 16),
                expansion_add=params.get("expansion_add", 128),
                expansion_search=params.get("expansion_search", 64),
            )

    def index(self, global_embeddings, global_idxs, partial_embeddings, partial_idxs):
        self.global_idxs.extend(global_idxs)
        self.partial_idx_2_global_idx.extend(partial_idxs)

        if self.use_exact:
            self.global_embeddings.extend(global_embeddings)
            self.partial_embeddings.extend(partial_embeddings)
        else:
            gkeys = np.asanyarray(global_idxs)
            gvects = np.asanyarray(global_embeddings)
            pkeys = np.arange(self.usearch_pkeys_count, self.usearch_pkeys_count + len(partial_idxs))
            self.usearch_pkeys_count += len(partial_idxs)  # move internal cursor for partial idxs

            pvects = np.asanyarray(partial_embeddings)

            self.global_index.add(gkeys, gvects)
            self.partial_index.add(pkeys, pvects)

    def query(
        self,
        global_query_embeddings,
        partial_query_embeddings,
        gk: int,
        pk: int,
        return_data: bool,
        queries: Sequence[Any],
        data: Sequence[Any] = [],
    ) -> ResultCollection:
        if return_data and not data:
            raise ValueError("Can't return data, data is empty")

        # Using USearch exact search
        if self.use_exact:

            # make sure we have contigious np.arrays
            self.global_embeddings = np.asanyarray(self.global_embeddings)
            self.partial_embeddings = np.asanyarray(self.partial_embeddings)

            # global embeddings
            glen = len(self.global_embeddings)
            gcount = gk if gk < glen else glen

            # partial embeddings
            plen = len(self.partial_embeddings)
            pcount = pk if pk < plen else plen

            gmatches_batch = search(
                self.global_embeddings,
                global_query_embeddings,
                metric=MetricKind.IP,
                count=gcount,
                exact=self.use_exact,
            )
            pmatches_batch = search(
                self.partial_embeddings,
                partial_query_embeddings,
                metric=MetricKind.IP,
                count=pcount,
                exact=self.use_exact,
            )

        # use USearch ANN search
        else:
            # global search
            glen = self.global_index.size
            gcount = gk if gk < glen else glen
            gmatches_batch = self.global_index.search(global_query_embeddings, count=gcount)

            # partial search
            plen = self.partial_index.size
            pcount = pk if pk < plen else plen
            pmatches_batch = self.partial_index.search(partial_query_embeddings, count=pcount)

        # compute global matches
        results_map: Dict[str, Result] = {}
        matches_map: Dict[str, Dict[str, Match]] = defaultdict(dict)
        for query_idx, gmatches in enumerate(gmatches_batch):
            result = Result(query_idx=query_idx)
            if return_data:
                result.query = queries[query_idx]

            for rank, m in enumerate(gmatches):
                target_idx = m.key
                similarity = 1 - m.distance

                match = Match(idx=target_idx, global_rank=rank, global_similarity=similarity)

                if return_data:
                    match.data = data[target_idx]

                # is a match?
                if similarity >= self.global_threshold:
                    match.is_global_match = True
                    result.num_global_matches += 1

                # save the match in a map to find it back in partial analysis
                matches_map[query_idx][match.idx] = match

            # save in a map to find it back during partial analysis
            results_map[result.query_idx] = result

        # compute partial matches
        for query_idx, pmatches in enumerate(pmatches_batch):
            result = results_map[query_idx]

            for rank, m in enumerate(pmatches):
                target_idx = self.partial_idx_2_global_idx[m.key]
                similarity = 1 - m.distance

                if result.query_idx in matches_map and target_idx in matches_map[result.query_idx]:
                    # if match exists update it
                    match = matches_map[result.query_idx][target_idx]
                    match.partial_rank = rank
                    match.partial_similarity = similarity

                else:
                    # if match does not exist create it
                    match = Match(idx=target_idx, partial_rank=rank, partial_similarity=similarity)
                    if return_data:
                        match.data = data[target_idx]

                # check if it is a match
                if similarity >= self.partial_threshold:
                    match.is_partial_match = True
                    result.num_partial_matches += 1

                # add to the matches map
                matches_map[result.query_idx][match.idx] = match

            # add back to result map
            results_map[result.query_idx] = result

        # create and return results collection
        results_collection = ResultCollection()
        for result in results_map.values():
            matches = [m for m in matches_map[result.query_idx].values()]
            result.matches = matches
            results_collection.total_global_matches += result.num_global_matches  # noqa
            results_collection.total_partial_matches += result.num_partial_matches  # noqa
            results_collection.results.append(result)
        return results_collection

    def reset(self) -> bool:
        if self.use_exact:
            self.global_embeddings = []
            self.partial_embeddings = []
        else:
            self.global_index.reset()
            self.partial_index.reset()

        self.global_idxs = []
        self.partial_idx_2_global_idx = []
        self.usearch_pkeys_count = 0

    def save(self, path: str) -> bool:
        raise NotImplementedError

    def load(self, path: str) -> bool:
        raise NotImplementedError
