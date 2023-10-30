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
from typing import AnyStr, Sequence, Tuple

import numpy as np

from ...config import floatx
from ...types import BatchGlobalEmbeddings, BatchPartialEmbeddings
from ..embedder import Embedder
from .binarizer import binarizer


class TextEmbedder(Embedder):
    """Convert text to embeddings

    Use RetSim model to convert text to embeddings
    """

    def __init__(self, batch_size: int, model_id: str = "text/retsim/v1", verbose: int = 0) -> None:
        # model loading is handled in the super
        super().__init__(batch_size=batch_size, model_id=model_id, verbose=verbose)

        # TODO (marinazh): change to be dependent on model, too brittle
        self._chunk_size = 512
        self._embedding_size = 256

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def chunk_size(self):
        return self._chunk_size

    def embed(
        self,
        inputs: Sequence[AnyStr],
    ) -> Tuple[BatchGlobalEmbeddings, BatchPartialEmbeddings]:
        """Compute text embeddings.

        A note on performance:
        inputs are non constant size so there is a lot vectorization ops
        we can't use or have to do by hand. That said we are in O(N)
        complexity. There is headroom to go faster if bottleneck.

        Args:
            inputs (Sequence[AnyStr]): _description_

        Returns:
            Tuple[BatchGlobalEmbeddings, BatchPartialEmbeddings]: _description_
        """
        batch, docids = binarizer(inputs, chunk_size=self.chunk_size)
        partial_embeddings = self.predict(batch)

        # averaging - might be faster with TF FIXME: try and benchmark
        partial_embeddings = np.asanyarray(partial_embeddings, dtype=floatx())
        global_embeddings = []
        stacked_partial_embeddings = []  # we need those
        for idxs in docids:
            embs = np.take(partial_embeddings, idxs, axis=0)
            avg_emb = np.sum(embs, axis=0) / len(embs)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            global_embeddings.append(avg_emb)
            stacked_partial_embeddings.append(embs)

        # NOTE: avg_embeddings is a list of np.array here. We need to stack later.
        return global_embeddings, stacked_partial_embeddings
