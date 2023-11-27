# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Sequence

import numpy as np

from ...types import BatchEmbeddings
from ..embedder import Embedder
from .binarizer import binarizer


class TextEmbedder(Embedder):
    """Converts text to embeddings using the RETSim text embedding model.

    The RETSim model embeds texts into 256-dimensional float vectors. The embeddings
    are trained using TensorFlow Similarity to be robust against adversarial attacks.
    The model is able to handle inputs of arbitrary length.
    """

    def __init__(self, batch_size: int, model_id: str = "text/retsim/v1", verbose: int = 0):
        """Initialize a TextEmbedder.

        Args:
            batch_size: Batch size for inference.

            model_id: ID of model, defaults to `text/retsim/v1`.

            verbose: Verbosity mode.
        """
        # model loading is handled in the super
        super().__init__(batch_size=batch_size, model_id=model_id, verbose=verbose)
        self._chunk_size = 512
        self._embedding_size = 256

    @property
    def embedding_size(self) -> int:
        """Returns the embedding size of the text embedding model."""
        return self._embedding_size

    def embed(
        self,
        inputs: Sequence[str],
    ) -> BatchEmbeddings:
        """Compute embeddings for a batch of text inputs.

        Args:
            inputs: Input texts to embed.

        Returns:
            Text embeddings for each text in `inputs`.
        """
        batch, docids = binarizer(inputs, chunk_size=self._chunk_size)  # noqa
        partial_embeddings = self.predict(batch)
        partial_embeddings = np.asanyarray(partial_embeddings)

        # average partial embeddings together into global embeddings
        global_embeddings = []
        for idxs in docids:
            embs = np.take(partial_embeddings, idxs, axis=0)

            # averaging partial embeddings together, if needed
            if len(embs) > 1:
                avg_emb = np.sum(embs, axis=0) / len(embs)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            else:
                avg_emb = embs[0]  # (1, 256) shape

            global_embeddings.append(avg_emb)

        return global_embeddings
