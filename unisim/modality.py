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
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from . import backend as B
from .dataclass import Result, ResultCollection
from .embedder import Embedder
from .enums import IndexerType
from .indexer import Indexer
from .types import BatchGlobalEmbeddings, BatchPartialEmbeddings


# put that as an abstract class
class Modality(object):
    """Wrapper class that allows to manipulate various data modality in a
    generic way using various type of indexer.

    Always use: np.asanyarray() when making sure everything is properly casted.
    Don't use np.array or np.asarray()
    """

    def __init__(
        self,
        store_data: bool,
        global_threshold: float,
        partial_threshold: float,
        index_type: str | IndexerType,
        batch_size: int,
        embedding_model: Embedder,
        model_id: str,
        index_params: Dict[str, Any] | None = None,
        verbose: int = 0,
    ) -> None:
        # model
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.model_id = model_id
        self.global_threshold = global_threshold
        self.partial_threshold = partial_threshold

        # indexes
        self.global_index_size = 0  # track idxs as we have two indexers
        self.index_type = index_type
        self.index_params = index_params if index_params else {}
        self.store_data = store_data

        # internal state
        self.is_initialized: bool = False
        self.verbose = verbose
        self.indexed_data = []

    def _lazy_init(self):
        "Lazily init models and indexers"
        # don't initialize twice
        if self.is_initialized:
            return

        # embedder
        self.embedding_size = self.embedder.embdding_size
        self.partial_size = self.embedder.chunk_size

        # indexer
        self.indexer = Indexer(
            embedding_size=self.embedding_size,
            use_tf_knn=self.use_tf_knn,
            index_type=self.index_type,
            global_threshold=self.global_threshold,
            partial_threshold=self.partial_threshold,
            params=self.index_params,
        )
        self.is_initialized = True

    # direct embedding manipulation
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

        ges_results = np.concatenate(ges_results, axis=0)
        pes_results = np.concatenate(pes_results, axis=0)
        return ges_results, pes_results

    def similarity(self, input1: Any, input2: Any) -> float:
        # compute embeddings
        batch = [input1, input2]
        ge, pe = self.embed(batch)

        # global distance
        similarity = B.cosine_similarity(ge[0], ge[1])
        similarity = np.asanyarray(similarity)[0]

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
            fpes, pes_idxs = self._flatten_partial_embeddings(bpes, idxs)

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

            fpqe, _ = self._flatten_partial_embeddings(pqe)

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

    def match(self, input) -> Result:
        raise NotImplementedError

    # dedup
    def dedup(self, inputs: Sequence[Any]):
        raise NotImplementedError

    # persistence
    def save(self, filepath):
        # ! DON't FORGET TO save inputs or move it to rockdb
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def _flatten_partial_embeddings(
        self,
        batch_partial_embeddings,
        ges_idxs: Sequence[int] | None = None,
    ):
        "Flatten partial embeddings and remap idxs to global ones"
        if ges_idxs is None:
            ges_idxs = []
        flatten_pes = []
        pes_idxs = []
        for idx, pes in enumerate(batch_partial_embeddings):
            for pe in pes:
                flatten_pes.append(pe)
                if ges_idxs:
                    pes_idxs.append(ges_idxs[idx])
        return np.asanyarray(flatten_pes), pes_idxs
