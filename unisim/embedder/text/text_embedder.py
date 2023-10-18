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

    def __init__(self, batch_size: int, version: int = 1, verbose: int = 0) -> None:
        # model loading is handled in the super
        super().__init__(batch_size=batch_size, modality="text", model_version=version, verbose=verbose)
        # set constanstant
        self.chunk_size = 512
        self.embdding_size = 256

    def embed(
        self,
        inputs: Sequence[AnyStr],
    ) -> Tuple[BatchGlobalEmbeddings, BatchPartialEmbeddings]:
        """Compute text embeddins

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
        avg_embeddings = []
        stacked_partial_embeddings = []  # we need those
        for idxs in docids:
            embs = np.take(partial_embeddings, idxs, axis=0)
            avg_emb = np.sum(embs, axis=0) / len(embs)
            avg_embeddings.append(avg_emb)
            stacked_partial_embeddings.append(embs)

        # NOTE: avg_embeddings is a list of np.array here. We need to stack later.
        return avg_embeddings, stacked_partial_embeddings
