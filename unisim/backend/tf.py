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
from typing import Any, Sequence

import tensorflow as tf
from tensorflow.keras import Model

# typing
from ..types import BatchDistances2D, BatchEmbeddings


def cosine_similarity(query_embeddings: BatchEmbeddings, index_embeddings: BatchEmbeddings) -> BatchDistances2D:
    """Compute cosine similarity between embeddings

    Args:
        query_embeddings: embeddings of the content to be searched
        index_embeddings: embeddings of the indexed content
    Returns:
        distances: matrix of distances
    """
    return tf.matmul(query_embeddings, index_embeddings, transpose_b=True)


def load_model(path: Path, verbose: int = 0):
    # specialize path
    mpath = str(path)
    if verbose:
        print(f"|-model path: {mpath}")

    model: tf.keras.Model = tf.keras.models.load_model(mpath)
    if verbose:
        model.summary()
    return model


def predict(model: Model, batch: Sequence[Any]) -> BatchEmbeddings:
    # add strategy here
    return model(batch, training=False)
