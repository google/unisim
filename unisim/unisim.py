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

            params: Additional parameters to be passed into USearch Index,
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
            print("UniSim is storing a copy of the indexed data")
            print("If you are using large data corpus, consider disabling this behavior using store_data=False")
        else:
            print("UniSim is not storing a copy of the indexed data to save memory")
            print("If you want to store a copy of the data, use store_data=True")

        if use_accelerator and get_accelerator() == AcceleratorType.cpu:
            print("Accelerator is not available, using cpu instead")
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
        """Compute embeddings for a batch of inputs.

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

    def search(self, inputs: Sequence[Any], similarity_threshold: float, k: int = 1) -> ResultCollection:
        """Search for and return the k closest matches for a set of queries,
        and mark the ones that are closer than `similarity_threshold` as
        near-duplicate matches.

        Args:
            inputs: Query inputs for the search.

            similarity_threshold: Similarity threshold for near-duplicate
                match, where a query and a search result are considered
                near-duplicate matches if their similarity is higher than
                `similarity_threshold`.

            k: Number of nearest neighbors to lookup.

        Returns
            result_collection: ResultCollection containing the search results.
        """
        results = ResultCollection()
        for b_offset in range(0, len(inputs), self.batch_size):
            batch = inputs[b_offset : b_offset + self.batch_size]
            embs = self.embed(batch)

            r = self.indexer.search(
                queries=batch,
                query_embeddings=embs,
                similarity_threshold=similarity_threshold,
                k=k,
                return_data=self.store_data,
                return_embeddings=self.return_embeddings,
                data=self.indexed_data,
            )
            results.merge_result_collection(r)

        return results

    def match(self, queries: Sequence[Any], targets: Sequence[Any]) -> DataFrame:
        """Find the closest matches for queries in a list of targets and
        return the result as a pandas DataFrame.

        Args:
            queries: Input queries to search for.

            targets: Targets to search in, e.g. for each query, find the nearest
                match in `targets`.

        Returns:
            Returns a pandas DataFrame with ["Query", "Match", "Similarity"]
            columns, representing each query, nearest match in `targets`, and
            their similarity value.
        """
        # add all targets
        self.add(targets)

        # search all queries, so it doesn't depend on similarity threshold
        results = self.search(queries, similarity_threshold=0.0, k=1).results

        # create pandas df
        data = []
        for i in range(len(results)):
            query = queries[i]
            match = results[i].matches[0]
            similarity = match.similarity
            matched = targets[match.idx]
            data.append([query, matched, similarity])

        df = DataFrame(data, columns=["Query", "Match", "Similarity"])
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
