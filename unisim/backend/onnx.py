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
from typing import Any, Dict, Sequence, Union

import numpy as np
from onnxruntime import InferenceSession

from ..types import BatchEmbeddings


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


# ONNX
_providers = [
    ("CUDAExecutionProvider", {"enable_cuda_graph": "1"}),
    "DmlExecutionProvider",  # directml windows
    "CPUExecutionProvider",
]


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
