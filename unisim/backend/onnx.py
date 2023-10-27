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

import numpy as np
from pathlib import Path
from onnxruntime import InferenceSession

# typing
from ..types import BatchEmbeddings
from ..types import BatchDistances2D


def cosine_similarity(query_embeddings: BatchEmbeddings, index_embeddings: BatchEmbeddings) -> BatchDistances2D:
    """Compute cosine similarity between embeddings

    Args:
        query_embeddings: embeddings of the content to be searched
        index_embeddings: embeddings of the indexed content
    Returns:
        distances: matrix of distances
    """
    return np.dot(query_embeddings, index_embeddings.T)


# ONNX
_providers = [
    ("CUDAExecutionProvider", {"enable_cuda_graph": "1"}),
    "DmlExecutionProvider",  # directml windows
    "CPUExecutionProvider",
]


def load_model(path: Path, verbose: int = 0):
    # specialize path
    mpath = str(path) + ".onnx"
    if verbose:
        print(f"|-model path: {mpath}")

    sess = InferenceSession(mpath, providers=_providers)

    # getting input/output info dynamically
    model = {
        "sess": sess,
        "input_name": sess.get_inputs()[0].name,
        "input_shape": sess.get_inputs()[0].shape,
        "input_type": sess.get_inputs()[0].type,
        "output_name": sess.get_outputs()[0].name,
        "output_shape": sess.get_outputs()[0].shape,
        "output_type": sess.get_outputs()[0].type,
    }

    if verbose:
        print(f'|-input: {model["input_shape"]}, {model["input_type"]}')
        print(f'|-output: {model["output_shape"]}, {model["output_type"]}')

    return model


def predict(model, batch) -> BatchEmbeddings:
    out = model["sess"].run([model["output_name"]], {model["input_name"]: batch})
    return np.asanyarray(out[0])
