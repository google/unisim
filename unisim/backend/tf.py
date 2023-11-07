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

from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model

# typing
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
