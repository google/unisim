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
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from . import backend as B
from .config import get_accelerator
from .dataclass import ResultCollection
from .embedder import Embedder
from .enums import AcceleratorType, IndexerType
from .indexer import Indexer
from .types import BatchGlobalEmbeddings, BatchPartialEmbeddings
from .utils import flatten_partial_embeddings, get_batched_inputs


class UniSim(ABC):

    def __init__(
        self,
        store_data: bool,
        global_threshold: float,
        partial_threshold: float,
        index_type: str | IndexerType,
        batch_size: int,
        use_accelerator: bool,
        model_id: str,
        embedder: Embedder,
        index_params: Dict[str, Any] | None = None,
        verbose: int = 0,
    ) -> None:
        self.store_data = store_data
        self.global_threshold = global_threshold
        self.partial_threshold = partial_threshold
        self.index_type = index_type if isinstance(index_type, IndexerType) else IndexerType[index_type]
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
        self.global_index_size = 0  # track idxs as we have two indexers
        self.is_initialized: bool = False
        self.indexed_data = []

    def _lazy_init(self):
        "Lazily init models and indexers"
        # check we don't initialize indexer twice
        if self.is_initialized:
            return

        # initialize indexer
        self.embedding_size = self.embedder.embedding_size
        self.indexer = Indexer(
            embedding_size=self.embedding_size,
            index_type=self.index_type,
            global_threshold=self.global_threshold,
            partial_threshold=self.partial_threshold,
            params=self.index_params,
        )
        self.is_initialized = True

    def embed(
        self,
        inputs: Sequence[Any],
    ) -> Tuple[BatchGlobalEmbeddings, BatchPartialEmbeddings]:
        self._lazy_init()
        ges_results = []
        pes_results = []
        for b_offset in range(0, len(inputs), self.batch_size):
            batch = inputs[b_offset : b_offset + self.batch_size]
            batch = np.asanyarray(batch)
            ges, pes = self.embedder.embed(inputs=batch)
            ges_results.append(ges)
            pes_results.append(pes)

        # TODO (marinazh): L2 normalize
        ges_results = np.concatenate(ges_results, axis=0)
        pes_results = np.concatenate(pes_results, axis=0)
        return ges_results, pes_results

    def similarity(self, input1: Any, input2: Any) -> float:
        # compute embeddings
        batch = [input1, input2]
        ge, _ = self.embed(batch)

        # compute global similarity
        similarity = B.cosine_similarity(ge, ge)
        similarity = np.asanyarray(similarity[0][1])

        # clip sometimes for floating point error
        similarity = np.clip(similarity, 0, 1)

        return similarity

    # indexing
    def index(self, inputs: Sequence[Any]) -> Sequence[int]:
        ges_idxs = []
        for b_offset in range(0, len(inputs), self.batch_size):
            batch = inputs[b_offset : b_offset + self.batch_size]
            ges, bpes = self.embed(batch)

            # compute the new global idxs
            idxs = [i + self.global_index_size for i in range(len(ges))]
            self.global_index_size += len(idxs)

            # flatten partial embeddings and maps them to global idxs
            fpes, pes_idxs = flatten_partial_embeddings(bpes, idxs)

            # indexing global and partials
            self.indexer.index(ges, idxs, fpes, pes_idxs)

            # store inputs if requested
            if self.store_data:
                self.indexed_data.extend(batch)

            # store the global idxs
            ges_idxs.extend(idxs)
        return ges_idxs

    def search(self, inputs: Sequence[Any], gk: int = 5, pk: int = 5):
        results = ResultCollection()
        for b_offset in range(0, len(inputs), self.batch_size):
            batch = inputs[b_offset : b_offset + self.batch_size]
            gqe, pqe = self.embed(batch)

            fpqe, _ = flatten_partial_embeddings(pqe)

            r = self.indexer.query(
                global_query_embeddings=gqe,
                partial_query_embeddings=fpqe,
                gk=gk,
                pk=pk,
                return_data=self.store_data,
                queries=batch,
                data=self.indexed_data,
            )
            results.merge_result_collection(r)

        return results

    def reset_index(self):
        self._lazy_init()
        self.global_index_size = 0
        self.indexer.reset()
        self.indexed_data = []

    def info(self):
        self._lazy_init()

        print("[Embedder]")
        print(f"|-batch_size: {self.batch_size}")
        print(f"|-model_id: {self.model_id}")
        print(f"|-embedding_size: {self.embedding_size}")

        print("[Indexer]")
        print(f"|-index_type: {self.index_type.name}")
        print(f"|-use_accelerator: {self.use_accelerator}")
        print(f"|-store index data: {self.store_data}")

    def dedup(self, inputs: Sequence[Any]):
        raise NotImplementedError

    def save(self, filepath: str):
        # ! DON't FORGET TO save inputs or move it to rockdb
        raise NotImplementedError

    def load(self, filepath: str):
        raise NotImplementedError
