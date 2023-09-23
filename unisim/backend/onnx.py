import numpy as np
from pathlib import Path
from onnxruntime import InferenceSession

# typing
from ..types import TensorEmbedding, BatchEmbeddings
from ..types import GlobalEmbedding, BatchGlobalEmbeddings  # noqa
from ..types import PartialEmbedding, PartialEmbeddings, BatchPartialEmbeddings  # noqa
from ..types import BatchDistances, BatchDistances2D, BatchIndexes
from typing import Tuple


def average_embeddings(embeddings: PartialEmbeddings,
                       axis: int = 1) -> TensorEmbedding:
    "Compute embedding average"
    raise NotImplementedError('copy tf when done')
    return np.sum(embeddings, axis=axis) / embeddings.shape[axis]


def cosine_similarity(query_embeddings: BatchEmbeddings,
                      index_embeddings: BatchEmbeddings) -> BatchDistances2D:
    """Compute cosine similarity between embeddings

    Args:
        query_embeddings: embeddings of the content to be searched
        index_embeddings: embeddings of the indexed content
    Returns:
        distances: matrix of distances
    """
    return np.dot(query_embeddings, index_embeddings.T)


# ONNX
_model: InferenceSession
_providers = [("CUDAExecutionProvider", {"enable_cuda_graph": '1'}),
              'DmlExecutionProvider',  # directml windows
              'CPUExecutionProvider']


def load_model(path: Path, verbose: int = 0):
    # specialize path
    mpath = str(path) + '.onnx'
    if verbose:
        print(f"|-model path: {mpath}")

    sess = InferenceSession(mpath, providers=_providers)
    model = {
        "sess": sess,
        # getting input/output info dynamically
        "input_name": sess.get_inputs()[0].name,
        "input_shape": sess.get_inputs()[0].shape,
        "input_type": sess.get_inputs()[0].type,
        "output_name": sess.get_outputs()[0].name,
        "output_shape": sess.get_outputs()[0].shape,
        "output_type": sess.get_outputs()[0].type

    }
    if verbose:
        print(f'|-input: {model["input_shape"]}, {model["input_type"]}')
        print(f'|-output: {model["output_shape"]}, {model["output_type"]}')
    return model


def predict(model, batch) -> BatchEmbeddings:
    out = model['sess'].run([model['output_name']],
                            {model['input_name']: batch})
    return out
