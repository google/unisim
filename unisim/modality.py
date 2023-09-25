import numpy as np
from perfcounters import PerfCounters

from . import backend as B
from .embedder import Embedder, TextEmbedder
from .indexer import Indexer
from .enums import ModalityType, IndexerType
from .dataclass import Result, Similarity, Match

from .types import GlobalEmbedding, PartialEmbeddings
from .types import BatchGlobalEmbeddings, BatchPartialEmbeddings
from typing import Sequence, Any, Tuple, Dict


# put that as an abstract class
class Modality(object):
    """Wrapper class that allows to manipulate various data modality in a
    generic way using various type of indexer.

    Always use: np.asanyarray() when making sure everything is properly casted.
    Don't use np.array or np.asarray()
    """

    def __init__(self,
                 global_threshold: float,
                 partial_threshold: float,
                 modality: ModalityType,
                 model_version: int,
                 indexer_type: IndexerType,
                 indexer_params: Dict,
                 use_tf_knn: bool,
                 store_data: bool,
                 verbose: int = 0) -> None:

        # model
        self.modality = modality
        self.model_version = model_version
        self.global_threshold = global_threshold
        self.partial_threshold = partial_threshold

        # indexes
        self.index_idxs = 0  # track idxs as we have two indexers
        self.indexer_type = indexer_type
        self.indexer_params = indexer_params
        self.use_tf_knn = use_tf_knn
        self.store_data = store_data

        # internal state
        self.is_initialized: bool = False
        self.verbose = verbose
        self.indexed_data = []

    def _lazy_init(self):
        "Lazily init models and indexers"
        # don't initialize twice
        if self.is_initialized:
            return

        # embedder
        if self.modality == ModalityType.text:
            self.embedder = TextEmbedder(version=self.model_version,
                                         verbose=self.verbose)
            self.embedding_size = self.embedder.embdding_size
            self.partial_size = self.embedder.chunk_size
        else:
            raise ValueError(f'Unknown modality: {self.modality}')

        # indexer
        self.indexer = Indexer(embedding_size=self.embedding_size,
                               use_tf_knn=self.use_tf_knn,
                               index_type=self.indexer_type,
                               global_threshold=self.global_threshold,
                               partial_threshold=self.partial_threshold,
                               params=self.indexer_params)
        self.is_initialized = True

    # direct embedding manipulation
    def embed(self, input) -> Tuple[GlobalEmbedding, PartialEmbeddings]:
        box = [input]
        ge, pe = self.batch_embed(box)
        return ge[0], pe[0]

    def batch_embed(self,
                    inputs: Sequence[Any],
                    verbose: int = 0) -> Tuple[BatchGlobalEmbeddings,
                                               BatchPartialEmbeddings]:
        self._lazy_init()
        inputs = np.asanyarray(inputs)
        ges, pes = self.embedder.batch_compute_embeddings(inputs,
                                                          verbose=verbose)
        return ges, pes

    # fixme: return a match or similarity object
    def similarity(self, input1, input2) -> Similarity:

        # compute embeddings
        batch = [input1, input2]
        ge, pe = self.batch_embed(batch)

        # global distance
        global_distances = B.cosine_similarity(ge, ge)
        global_distances = np.asanyarray(global_distances)

        # init similarity dataclass
        simres = Similarity(
            query_embedding=ge[0],
            target_embedding=ge[1],
            distance=float(global_distances[0][1])  # we don't want the diag
        )

        # is it a match
        if simres.distance >= self.global_threshold:
            simres.is_global_match = True

        # partial matches
        partial_distances = B.cosine_similarity(pe[0], pe[1])

        # FIXME use GPU acceleration if possible]
        partial_distances = np.asanyarray(partial_distances)
        for idx1, chunk_distances in enumerate(partial_distances):
            idx2 = np.argmax(chunk_distances)
            dist = chunk_distances[idx2]
            pmatch = Match(
                idx=0,
                global_rank=1,
                global_similarity=float(dist),
                match_len=self.partial_size,
                target_match_position=(idx2 + 1) * self.partial_size,
                query_match_position=(idx1 + 1) * self.partial_size)
            if dist > self.partial_threshold:
                pmatch.is_partial_match = True
                simres.is_partial_match = True
            simres.partial_matches.append(pmatch)
        return simres

    # indexing
    def index(self, input) -> int:
        inputs = [input]
        res = self.batch_index(inputs=inputs)
        return res[0]

    def batch_index(self, inputs, verbose: int = 0) -> Sequence[int]:
        cnts = PerfCounters()
        cnts.start('total')

        cnts.start('batch_embed')
        ges, bpes = self.batch_embed(inputs, verbose=verbose)
        cnts.stop('batch_embed')

        # compute the new global idxs
        cnts.start('compute_global_idxs')
        ges_idxs = [i + self.index_idxs for i in range(len(ges))]
        self.index_idxs += len(ges_idxs)
        cnts.stop('compute_global_idxs')

        # flatten partial embeddings and maps them to global idxs
        cnts.start('flatten_partial_embeddings')
        fpes, pes_idxs = self._flatten_partial_embeddings(bpes, ges_idxs)
        cnts.stop('flatten_partial_embeddings')

        # indexing global and partials
        cnts.start('batch_index')
        self.indexer.batch_index(ges, ges_idxs, fpes, pes_idxs)
        cnts.stop('batch_index')

        # store inputs if requested
        if self.store_data:
            cnts.start('store_data')
            self.indexed_data.extend(inputs)
            cnts.stop('store_data')
        cnts.stop('total')
        if verbose:
            cnts.report()
        return ges_idxs

    # direct search
    def search(self, input, k: int = 5):
        raise NotImplementedError

    def batch_search(self, inputs,
                     gk: int = 5, pk: int = 5):

        gqe, pqe = self.batch_embed(inputs)

        fpqe, _ = self._flatten_partial_embeddings(pqe)

        results = self.indexer.batch_query(global_query_embeddings=gqe,
                                           partial_query_embeddings=fpqe,
                                           gk=gk, pk=pk,
                                           return_data=self.store_data,
                                           queries=inputs,
                                           data=self.indexed_data)
        return results

    def reset_index(self):
        self._lazy_init()
        self.index_idxs = 0
        self.indexer.reset()
        self.indexed_data = []

    def match(self, input) -> Result:
        raise NotImplementedError

    # dedup
    def dedup(self, inputs: Sequence[Any]):
        raise NotImplementedError

    # persistence
    def save(self, filepath):
        # ! DON't FORGET TO save inputs or move it to rockdb
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def _flatten_partial_embeddings(self, batch_partial_embeddings,
                                    ges_idxs: Sequence[int] = []):
        "Flatten partial embeddings and remap idxs to global ones"
        flatten_pes = []
        pes_idxs = []
        for idx, pes in enumerate(batch_partial_embeddings):
            for pe in pes:
                flatten_pes.append(pe)
                if ges_idxs:
                    pes_idxs.append(ges_idxs[idx])
        return np.asanyarray(flatten_pes), pes_idxs
