import numpy as np
from ... import backend as B
from .binarizer import binarizer
from ..embedder import Embedder

from ...config import floatx
from typing import Sequence, AnyStr, Tuple
from ...types import BatchGlobalEmbeddings
from ...types import BatchPartialEmbeddings


class TextEmbedder(Embedder):
    """Convert text to embeddings

    Use RetSim model to convert text to embeddings
    """

    def __init__(self, version: int = 1,
                 verbose: int = 0) -> None:
        # model loading is handled in the super
        super().__init__(modality='text', model_version=version,
                         verbose=verbose)
        # set constanstant
        self.chunk_size = 512
        self.embdding_size = 256

    def batch_compute_embeddings(self, inputs: Sequence[AnyStr]
                                 ) -> Tuple[BatchGlobalEmbeddings,
                                            BatchPartialEmbeddings]:
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
        # inputs: [text1, text2, text3]
        batch, docids = binarizer(inputs, chunk_size=self.chunk_size)

        # batch: [num_flatten_chunks, 24]  docids for each chunks
        partial_embeddings = self.predict(batch)

        # [num_flatten_chunk, 256]
        # gather_avg don't work as it is irregular shape (different len)
        # avg_embeddings = B.gather_and_avg(partial_embeddings, docids)

        # averaging - might be faster with TF FIXME: try and benchmark
        partial_embeddings = np.asanyarray(partial_embeddings, dtype=floatx())
        avg_embeddings = []
        stacked_partial_embeddings = []  # we need those
        for idxs in docids:
            embs = np.take(partial_embeddings, idxs, axis=0)
            avg_emb = np.sum(embs, axis=0) / len(embs)
            avg_embeddings.append(avg_emb)
            stacked_partial_embeddings.append(embs)
        avg_embeddings = np.asanyarray(avg_embeddings, dtype=floatx())
        # avg_embedding = B.average_embeddings(partial_embeddings)
        return avg_embeddings, stacked_partial_embeddings
