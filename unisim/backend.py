# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
from onnxruntime import InferenceSession

from .types import BatchEmbeddings


def cosine_similarity(query_embeddings: BatchEmbeddings, index_embeddings: BatchEmbeddings) -> np.ndarray:
    """Compute cosine similarity between embeddings using numpy.

    Args:
        query_embeddings: Embeddings of the content to be searched.

        index_embeddings: Embeddings of the indexed content.

    Returns:
        Matrix of cosine similarity values.
    """
    similarity: np.ndarray = np.dot(query_embeddings, index_embeddings.T)
    return similarity


# ONNX providers -- prefer CUDA Execution Provider over CPU Execution Provider 
_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']


def load_model(path: Path, verbose: int = 0) -> Dict[str, Any]:
    """Helper function to load Onnx model.

    Args:
        path: Path to the saved .onnx model.

        verbose: Print model details if verbose.

    Returns:
        Dict containing Onnx model session, input and output.
    """
    # specialize path
    mpath = str(path.with_suffix(".onnx"))
    if verbose:
        print(f"|-model path: {mpath}")

    sess = InferenceSession(mpath, providers=_providers)

    # getting input/output info dynamically
    model_dict = {
        "sess": sess,
        "input_name": sess.get_inputs()[0].name,
        "input_shape": sess.get_inputs()[0].shape,
        "input_type": sess.get_inputs()[0].type,
        "output_name": sess.get_outputs()[0].name,
        "output_shape": sess.get_outputs()[0].shape,
        "output_type": sess.get_outputs()[0].type,
    }

    if verbose:
        print(f'|-input: {model_dict["input_shape"]}, {model_dict["input_type"]}')
        print(f'|-output: {model_dict["output_shape"]}, {model_dict["output_type"]}')

    return model_dict


def predict(model_dict: Dict[str, Any], batch: Union[Sequence[Any], np.ndarray]) -> BatchEmbeddings:
    """Predict using Onnx model dictionary loaded from load_model().

    Args:
        model_dict: Model loaded using load_model().

        batch: Batch of inputs to generate predictions.

    Returns:
        Batch of embeddings generated from inputs
    """
    out = model_dict["sess"].run([model_dict["output_name"]], {model_dict["input_name"]: batch})
    return np.asanyarray(out[0])
