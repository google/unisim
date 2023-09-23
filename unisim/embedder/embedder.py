from pathlib import Path
from abc import ABC, abstractmethod

from .. import backend as B

# typing
from typing import Tuple, Any, Sequence
from ..types import BatchGlobalEmbeddings
from ..types import BatchPartialEmbeddings, BatchEmbeddings


class Embedder(ABC):
    "Convert inputs to embeddings suitable for indexing"

    def __init__(self, modality: str, model_version: int,
                 verbose: int = 0) -> None:
        self.modality = modality
        self.model_version = model_version
        self.model_fname = f'unisim_{modality}_v{model_version}'

        # fixme load ondemand from the internet
        # find the model path
        fpath = Path(__file__)
        model_path = fpath.parent / 'models' / self.model_fname

        # load model
        if verbose:
            print('[Loading model]')
            print(f'|-modality:{modality}')
            print(f'|-version:{model_version}')

        # load model
        self.model = B.load_model(model_path, verbose=verbose)

    @abstractmethod
    def batch_compute_embeddings(self, inputs: Sequence[Any]
                                 ) -> Tuple[BatchGlobalEmbeddings,
                                            BatchPartialEmbeddings]:
        """Compute embeddings for a batch of inputs

        Args:
            input:s the list modality input to be embedded
        Returns:
            Tuple[Embedding, Embeddings]: Inputs global embeddings,
            Input chunk embeddings (num_inputs, embedding_dim)
        """
        raise NotImplementedError

    def predict(self, batch) -> BatchEmbeddings:
        "Run inference using the loaded model with the right framework"
        return B.predict(self.model, batch=batch)
