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

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from usearch.index import BatchMatches, Index, MetricKind, search

from .dataclass import Match, Result, ResultCollection
from .enums import IndexerType
from .types import BatchEmbeddings


class Indexer:
    """Indexing system for UniSim that uses Usearch
    (https://github.com/unum-cloud/usearch) for both exact and approximate
    nearest neighbor (ANN) search. This enables us to efficiently index and
    find the nearest neighbors of embedding vectors.

    We strongly recommend using ANN search if the size of the dataset you
    are indexing or searching over is large, since it will provide a
    substantial speed up. Using exact search guarantees that you return the
    correct nearest neighbors every single time.
    """

    def __init__(
        self,
        embedding_size: int,
        index_type: IndexerType,
        params: Dict[str, Any] = {},
    ):
        """Initialize Indexer for UniSim, which uses the USearch package.

        For more details on USearch, see https://github.com/unum-cloud/usearch.

        Args:
            embedding_size: Size of the embedding vectors to index.

            index_type: Indexer type, either `IndexerType.exact` for exact
                search or `IndexerType.approx` for ANN search.

            params: Additional parameters to be passed into USearch Index,
                only used when index_type is approx, for ANN search. Supported
                dict keys include {metric, dtype, connectivity, expansion_add,
                expansion_search}, please see the USearch Index documentation
                for more detail on what each parameter does.
        """
        self.embedding_size = embedding_size
        self.index_type = index_type
        self.params = params

        # determine if we use ANN or exact search for the index
        self.use_exact = True if index_type == IndexerType.exact else False

        if self.use_exact:
            # exact matching and we need to store the embeddings in memory
            self.embeddings: List = []

        else:
            # initializing the USearch ANN index
            self.index = Index(
                ndim=self.embedding_size,
                metric="ip",
                dtype=params.get("dtype", "f32"),
                connectivity=params.get("connectivity", 16),
                expansion_add=params.get("expansion_add", 128),
                expansion_search=params.get("expansion_search", 64),
            )
        self.idxs: List[int] = []

    def add(self, embeddings: BatchEmbeddings, idxs: List[int]):
        """Add a batch of embeddings to the indexer.

        Args:
            embeddings: Embeddings to add.

            idxs: Indices corresponding to the embeddings to add.
        """
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
        similarity_threshold: float,
        k: int,
        drop_closest_match: bool = False,
        return_data: bool = True,
        return_embeddings: bool = True,
        data: Sequence[Any] = [],
    ) -> ResultCollection:
        """Search for and return the k closest matches for a set of queries,
        and mark the ones that are closer than `similarity_threshold` as
        near-duplicate matches.

        Args:
            queries: Input query data.

            query_embeddings: Embeddings corresponding to input queries.

            similarity_threshold: Similarity threshold for near-duplicate match,
                where a query and a search result are considered near-duplicate matches
                if their similarity is higher than `similarity_threshold`.

            k: Number of nearest neighbors to lookup.

            drop_closest_match: If True, remove the closest match before returning
                results. This is used when search queries == indexed set, since
                each query's closest match will be itself if it was already added
                to the index.

            return_data: Whether to return data corresponding to search results.

            return_embeddings: Whether to return embeddings for search results.

            data: Input data to fetch search result data from.

        Returns
            ResultCollection containing the search results.
        """
        if return_data and not data:
            raise ValueError("Can't return data, data is empty")

        # we need to increase k by 1 if we are dropping the closest match
        if drop_closest_match:
            k += 1

        # Using USearch exact search
        if self.use_exact:
            # check how many items to return in case k is larger than index size
            index_size = len(self.embeddings)
            count = k if k < index_size else index_size

            matches_batch = search(
                np.asanyarray(self.embeddings),
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
        results_map: Dict[int, Result] = {}
        matches_map: Dict[int, Dict[int, Match]] = defaultdict(dict)

        # for single batch lookup Matches is returned instead of BatchMatches
        if not isinstance(matches_batch, BatchMatches):
            matches_batch = [matches_batch]

        for query_idx, matches in enumerate(matches_batch):
            result = Result(query_idx=query_idx)
            if return_data:
                result.query_data = queries[query_idx]

            if return_embeddings:
                result.query_embedding = query_embeddings[query_idx]

            for rank, m in enumerate(matches):

                # drop closest match if set and increase the ranking of matches
                if drop_closest_match:
                    if rank == 0:
                        continue
                    else:
                        rank -= 1

                target_idx = m.key
                similarity = 1 - m.distance
                embedding = None
                if return_embeddings:
                    embedding = self.embeddings[target_idx] if self.use_exact else self.index[target_idx]
                    embedding = np.squeeze(embedding)

                match = Match(idx=target_idx, rank=rank, similarity=similarity, embedding=embedding)

                if return_data:
                    match.data = data[target_idx]

                # is a match?
                if similarity >= similarity_threshold:
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

    def reset(self):
        """Reset the index."""
        if self.use_exact:
            self.embeddings = []
        else:
            self.index.reset()
        self.idxs = []

    def save(self, path: Path | str):
        """Save the index to disk.

        Args:
            path: directory to save the index
        """
        if self.use_exact:
            with open(self._save_pickle_path(path), "wb") as f:
                pickle.dump((self.embeddings, self.idxs), f)
        else:
            self.index.save(self._save_usearch_path(path))

    def load(self, path: Path | str):
        """Reload index data from save directory and recreate the index
        with the underlying data.

        Args:
            path: Directory where the indexer was saved.

        Returns:
            Initialized indexer from save directory.
        """
        if self.use_exact:
            with open(self._save_pickle_path(path), "rb") as f:
                data = pickle.load(f)
            self.embeddings = data[0]
            self.idxs = data[1]
        else:
            self.index.load(self._save_usearch_path(path))

    def _save_pickle_path(self, path):
        return Path(path) / "index.pickle"

    def _save_usearch_path(self, path):
        return Path(path) / "index.usearch"
