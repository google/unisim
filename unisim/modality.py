import numpy as np
from .enums import ModalityType, IndexerType
from .dataclass import Result, Similarity, PartialMatch
from .config import floatx
from . import backend as B
from .embedder import Embedder, TextEmbedder
from .types import GlobalEmbedding, PartialEmbeddings
from .types import BatchGlobalEmbeddings, BatchPartialEmbeddings
from typing import Sequence, Any, Tuple


# put that as an abstract class
class Modality(object):
    """Wrapper class that allows to manipulate various data modality in a
    generic way using various type of indexer.
    """

    embedder: Embedder

    def __init__(self,
                 global_threshold: float,
                 partial_threshold: float,
                 modality: ModalityType,
                 model_version: int,
                 verbose: int = 0) -> None:

        self.modality = modality
        self.model_version = model_version
        self.global_threshold = global_threshold
        self.partial_threshold = partial_threshold
        self.verbose = verbose

        # fixme
        # self.index
        # FIXME: rename to initalized to also init the indexer
        self.embedding_size = 0
        self.chunk_size = 0
        self.embeder_initialized = False

    # FIXME move to initialize to also do indexers - global and partial
    def _load_model(self):
        "Load needed model on demand"
        # don't initialize twice
        if self.embeder_initialized:
            return

        if self.modality == ModalityType.text:
            self.embedder = TextEmbedder(version=self.model_version,
                                         verbose=self.verbose)
            self.embedding_size = self.embedder.embdding_size
            self.chunk_size = self.embedder.chunk_size
        else:
            raise ValueError(f'Unknown modality: {self.modality}')
        self.embeder_initialized = True

    # direct embedding manipulation
    def embed(self, input) -> Tuple[GlobalEmbedding, PartialEmbeddings]:
        box = [input]
        ge, pe = self.batch_embed(box)
        return ge[0], pe[0]

    def batch_embed(self,
                    inputs: Sequence[Any]) -> Tuple[BatchGlobalEmbeddings,
                                                    BatchPartialEmbeddings]:
        self._load_model()  # lazy loading model

        global_embeddings, partial_embeddings = self.embedder.batch_compute_embeddings(inputs)
        return global_embeddings, partial_embeddings

    # fixme: return a match or similarity object
    def similarity(self, input1, input2) -> Similarity:

        # compute embeddings
        batch = [input1, input2]
        ge, pe = self.batch_embed(batch)

        # global distance
        global_distances = B.cosine_similarity(ge, ge)

        # init similarity dataclass
        simres = Similarity(
            query_embedding=np.array(ge[0]),
            target_embedding=np.array(ge[1]),
            distance=float(global_distances[0][1])  # we don't want the diag
        )

        # is it a match
        if simres.distance >= self.global_threshold:
            simres.is_near_duplicate = True
            simres.is_match = True

        # partial matches
        partial_distances = B.cosine_similarity(pe[0], pe[1])

        # FIXME use GPU acceleration if possible]
        partial_distances = np.array(partial_distances)
        for idx1, chunk_distances in enumerate(partial_distances):
            idx2 = np.argmax(chunk_distances)
            dist = chunk_distances[idx2]
            pmatch = PartialMatch(
                distance=float(dist),
                match_len=self.chunk_size,
                target_idx=0,
                target_chunk_idx=idx2,
                target_match_position=(idx2 + 1) * self.chunk_size,
                query_match_position=(idx1 + 1) * self.chunk_size)
            if dist > self.partial_threshold:
                pmatch.is_partial_match = True
                simres.is_partial_match = True
            simres.partial_matches.append(pmatch)
        return simres

    # direct search
    def search(self, input, k: int = 5) -> Sequence[Result]:
        raise NotImplementedError

    def match(self, input) -> Result:
        raise NotImplementedError

    # dedup
    def dedup(self, inputs: Sequence[Any]):
        raise NotImplementedError
