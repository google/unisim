# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
from abc import ABC
from typing import Any, Dict, List, Sequence

import numpy as np
from pandas import DataFrame

from . import backend as B
from .config import get_accelerator
from .dataclass import ResultCollection
from .embedder import Embedder
from .enums import AcceleratorType, IndexerType
from .indexer import Indexer
from .types import BatchEmbeddings


class UniSim(ABC):
    """Abstract base class for UniSim.

    UniSim is designed to find near-duplicates and similar matches for content
    (such as text, images, multi-modal inputs) using embedding models. The
    UniSim packages enables finding the similarity between two items, indexing
    and querying for similar items, and finding near-duplicates in a dataset.
    """

    def __init__(
        self,
        store_data: bool,
        index_type: str | IndexerType,
        return_embeddings: bool,
        batch_size: int,
        use_accelerator: bool,
        model_id: str,
        embedder: Embedder,
        index_params: Dict[str, Any] | None = None,
        verbose: int = 0,
    ):
        """Create UniSim to index, search, and find near-duplicate matches
        using an embedding model.

        Args:
            store_data: Whether to store a copy of the indexed data and return
                data when returning search results.

            index_type: Indexer type, either "exact" for exact search or
                "approx" for Approximate Nearest Neighbor (ANN) search.

            return_embeddings: Whether to return embeddings corresponding to each
                search result when calling UniSim.search().

            batch_size: Batch size for inference.

            use_accelerator: Whether to use an accelerator (GPU), if available.

            model_id: String id of the model.

            embedder: Embedder object which converts inputs into embeddings using
                a model.

            params: Additional parameters to be passed into the Indexer,
                only used when index_type is approx, for ANN search. Supported
                dict keys include {metric, dtype, connectivity, expansion_add,
                expansion_search}, please see the USearch Index documentation
                for more detail on what each parameter does.

            verbose: Verbosity level, set to 1 for verbose.
        """
        self.store_data = store_data
        self.index_type = index_type if isinstance(index_type, IndexerType) else IndexerType[index_type]
        self.return_embeddings = return_embeddings
        self.batch_size = batch_size
        self.use_accelerator = use_accelerator
        self.model_id = model_id
        self.embedder = embedder
        self.index_params = index_params if index_params else {}
        self.verbose = verbose

        if self.store_data:
            logging.info("UniSim is storing a copy of the indexed data")
            logging.info("If you are using large data corpus, consider disabling this behavior using store_data=False")
        else:
            logging.info("UniSim is not storing a copy of the indexed data to save memory")
            logging.info("If you want to store a copy of the data, use store_data=True")

        if use_accelerator and get_accelerator() == AcceleratorType.cpu:
            logging.info("Accelerator is not available, using CPU")
            self.use_accelerator = False

        # internal state
        self.index_size = 0
        self.is_initialized: bool = False
        self.indexed_data: List[Any] = []

    def _lazy_init(self):
        """Lazily init models and indexers."""
        # check we don't initialize indexer twice
        if self.is_initialized:
            return

        # initialize indexer
        self.embedding_size = self.embedder.embedding_size
        self.indexer = Indexer(
            embedding_size=self.embedding_size,
            index_type=self.index_type,
            params=self.index_params,
        )
        self.is_initialized = True

    def embed(
        self,
        inputs: Sequence[Any],
    ) -> BatchEmbeddings:
        """Compute embeddings for inputs.

        Args:
            inputs: Sequence of inputs to embed.

        Returns:
            Embeddings for each input in `inputs`.
        """
        self._lazy_init()
        embeddings = []
        for b_offset in range(0, len(inputs), self.batch_size):
            batch = inputs[b_offset : b_offset + self.batch_size]
            embs = self.embedder.embed(inputs=batch)
            embeddings.append(embs)

        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def similarity(self, input1: Any, input2: Any) -> float:
        """Return the similarity between two inputs as a float.

        Args:
            input1: First input.

            input2: Second input.

        Returns:
            Similarity between the two inputs between 0 and 1.
        """
        # compute embeddings
        batch = [input1, input2]
        embs = self.embed(batch)

        # compute similarity
        similarity = B.cosine_similarity(embs, embs)
        similarity = np.asanyarray(similarity[0][1])

        # clip sometimes for floating point error
        similarity = np.clip(similarity, 0, 1)

        return float(similarity)

    def add(self, inputs: Sequence[Any]) -> List[int]:
        """Add inputs to the index.

        Args:
            inputs: Inputs to embed and add to the index.

        Returns:
             idxs: Indices corresponding to the inputs added to the index,
             where idxs[0] is the idx for inputs[0] in the index.
        """
        inputs_idxs = []
        for b_offset in range(0, len(inputs), self.batch_size):
            batch = inputs[b_offset : b_offset + self.batch_size]
            embs = self.embed(batch)

            # compute the new idxs
            idxs = [i + self.index_size for i in range(len(embs))]
            self.index_size += len(idxs)

            # indexing embeddings
            self.indexer.add(embs, idxs)

            # store inputs if requested
            if self.store_data:
                self.indexed_data.extend(batch)

            # store the idxs corresponding to inputs
            inputs_idxs.extend(idxs)

        return inputs_idxs

    def search(
        self, queries: Sequence[Any], similarity_threshold: float, k: int = 1, drop_closest_match: bool = False
    ) -> ResultCollection:
        """Search for and return the k closest matches for a set of queries,
        and mark the ones that are closer than `similarity_threshold` as
        near-duplicate matches.

        Args:
            queries: Query inputs to search for.

            similarity_threshold: Similarity threshold for near-duplicate
                match, where a query and a search result are considered
                near-duplicate matches if their similarity is higher than
                `similarity_threshold`.

            k: Number of nearest neighbors to lookup for each query input.

            drop_closest_match: If True, remove the closest match before returning
                results. This is used when search queries == indexed set, since
                each query's closest match will be itself if it was already added
                to the index.

        Returns
            result_collection: ResultCollection containing the search results.
        """
        results = ResultCollection()
        for b_offset in range(0, len(queries), self.batch_size):
            batch = queries[b_offset : b_offset + self.batch_size]
            embs = self.embed(batch)

            r = self.indexer.search(
                queries=batch,
                query_embeddings=embs,
                similarity_threshold=similarity_threshold,
                k=k,
                drop_closest_match=drop_closest_match,
                return_data=self.store_data,
                return_embeddings=self.return_embeddings,
                data=self.indexed_data,
            )
            results.merge_result_collection(r)

        return results

    def match(self, queries: Sequence[Any],
              targets: Sequence[Any] | None = None,
              similarity_threshold: float = 0.9,
              as_pandas_df: bool = True) -> DataFrame | ResultCollection:
        """Find the closest matches for queries in a list of targets and
        return the result as a pandas DataFrame.

        Args:
            queries: Input queries to search for.

            targets: Targets to search in, e.g. for each query, find the nearest
                match in `targets`. If None, then `queries` is used as the
                targets as well and matches are computed within a single list.

            similarity_threshold: Similarity threshold for near-duplicate
                match, where a query and a search result are considered
                near-duplicate matches if their similarity is higher than
                `similarity_threshold`.

        Returns:
            Returns a pandas DataFrame with ["query", "target", "similarity", "is_match"]
            columns, representing each query, nearest match in `targets`, their similarity
            value, and whether or not they are a match (if their similarity >=
            `similarity_threshold`).
        """
        # reset index
        self.reset_index()

        # defaults if we have targets
        drop_closest_match = False
        k = 1

        # if we are matching within the same list, drop the closest match
        # since it will be the query itself
        if not targets:
            targets = queries
            drop_closest_match = True
            k = 2

        # add all targets
        self.add(targets)

        # search all queries, so it doesn't depend on similarity threshold
        results = self.search(queries, similarity_threshold=0.0, k=k, drop_closest_match=drop_closest_match).results

        # cleanup index
        self.reset_index()


        # return results as a raw ResultCollection or pandas DataFrame
        if not as_pandas_df:
            return results
        else:
            data = []
            for i in range(len(results)):
                query = queries[i]
                match = results[i].matches[0]
                similarity = match.similarity
                is_match = similarity and similarity >= similarity_threshold
                matched = targets[match.idx]
                data.append([query, matched, similarity, is_match])

            df = DataFrame(data, columns=["query", "target", "similarity", "is_match"])
            return df

    def reset_index(self):
        """Reset the index by removing all previously-indexed data."""
        self._lazy_init()
        self.index_size = 0
        self.indexer.reset()
        self.indexed_data = []

    def info(self):
        """Display UniSim info."""
        self._lazy_init()

        print("[Embedder]")
        print(f"|-batch_size: {self.batch_size}")
        print(f"|-model_id: {self.model_id}")
        print(f"|-embedding_size: {self.embedding_size}")

        print("[Indexer]")
        print(f"|-index_type: {self.index_type.name}")
        print(f"|-use_accelerator: {self.use_accelerator}")
        print(f"|-store index data: {self.store_data}")
        print(f"|-return embeddings: {self.return_embeddings}")
