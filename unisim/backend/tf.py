# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model

from ..types import BatchEmbeddings


def cosine_similarity(query_embeddings: BatchEmbeddings, index_embeddings: BatchEmbeddings) -> Tensor:
    """Compute cosine similarity between embeddings using TensorFlow.

    Args:
        query_embeddings: Embeddings of the content to be searched.

        index_embeddings: Embeddings of the indexed content.

    Returns:
        Matrix of cosine similarity values.
    """
    return tf.matmul(query_embeddings, index_embeddings, transpose_b=True)


def load_model(path: Path, verbose: int = 0):
    """Helper function to load TensorFlow/Keras model.

    Args:
        path: Path to the saved model.

        verbose: Print model summary if verbose.

    Returns:
        tf.keras.Model reloaded from path.
    """
    # specialize path
    mpath = str(path)
    if verbose:
        print(f"|-model path: {mpath}")

    model: tf.keras.Model = tf.keras.models.load_model(mpath)

    if verbose:
        model.summary()
    return model


def predict(model: Model, batch: Union[Sequence[Any], np.ndarray]) -> BatchEmbeddings:
    """Predict using TensorFlow model.

    Args:
        model: TensorFlow/Keras model.

        batch: Batch of inputs to generate predictions.

    Returns:
        Batch of embeddings generated from inputs.
    """
    return model(batch, training=False)
