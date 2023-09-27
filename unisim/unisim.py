from abc import ABC
from .enums import IndexerType, ModalityType, AcceleratorType, BackendType
from .modality import Modality
from .viz import Visualization
from .config import get_accelerator, get_backend
from typing import Dict


class UniSim(ABC):

    # modality
    text: Modality
    # FIXME: add image when ready
    image: Modality
    multimodal: Modality

    # visulization
    viz: Visualization

    def __init__(self,
                 index_type: IndexerType,
                 indexer_params: Dict,
                 store_data: bool,
                 # Potentially use a diff batchsize by modality
                 batch_size: int = 128,
                 text_global_threshold: float = 0.9,
                 text_partial_threshold: float = 0.85,
                 text_model_version: int = 1,
                 use_tf_knn: bool = False,
                 verbose: int = 0) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.verbose = verbose
        self.index_type = index_type
        self.store_data = store_data

        if self.store_data:
            print("UniSim is storing a copy of the indexed data")
            print("if you are using large data corpus consider disable this behavior using store_data=False")
        else:
            print("UniSim is not storing a copy of the index data to save memory")
            print("If you want to store it use store_data=True")
        # params
        self.is_gpu = True if get_accelerator() == AcceleratorType.gpu else False  # noqa
        self.use_exact = True if index_type == IndexerType.exact else False
        self.use_tf = True if get_backend() == BackendType.tf else False

        # is tf_knn is forced by the user
        if use_tf_knn:
            self.use_tf_knn = True
        # else smart detection
        elif self.use_exact and self.use_tf and self.is_gpu:
            self.use_tf_knn = True
        else:
            self.use_tf_knn = False

        # viz
        self.viz = Visualization()

        # Initalizing the models/embedders
        self.text = Modality(batch_size=batch_size,
                             global_threshold=text_global_threshold,
                             partial_threshold=text_partial_threshold,
                             modality=ModalityType.text,
                             model_version=text_model_version,
                             indexer_type=index_type,
                             store_data=store_data,
                             indexer_params=indexer_params,
                             use_tf_knn=self.use_tf_knn,
                             verbose=verbose)

    def info(self):
        # FIXME more information e.g backend gpu etc
        print(f'[Embedder]')
        print(f'|-batch_size:{self.batch_size}')
        print("[Indexer]")
        print(f'|-is_exact:{self.use_exact}')
        print(f'|-use_tf_knn:{self.use_tf_knn}')
        print(f'|-store index data:{self.store_data}')


# add inmemory which don't have save
class ExactUniSim(UniSim):
    """Instanciate UniSim to perform exact matching

    Exact matching guarantee that you get exactly the right results at the cost
    of requiring N^2 computation. Works well in particular
    with a GPU sub 1M points.

    """
    def __init__(self,
                 batch_size: int = 128,
                 store_data: bool = True,
                 text_global_threshold: float = 0.9,
                 text_partial_threshold: float = 0.85,
                 text_model_version: int = 1,
                 use_tf_knn: bool = False,
                 verbose: int = 0) -> None:

        super().__init__(batch_size=batch_size,
                         index_type=IndexerType.exact,
                         indexer_params={},
                         store_data=store_data,
                         use_tf_knn=use_tf_knn,
                         text_global_threshold=text_global_threshold,
                         text_partial_threshold=text_partial_threshold,
                         text_model_version=text_model_version,
                         verbose=verbose)


class ApproxUniSim(UniSim):
    """Instanciate UniSim to perform approximate matching

    Ideal for fast matching large datasets

    """
    def __init__(self,
                 batch_size: int = 128,
                 store_data: bool = False,
                 text_global_threshold: float = 0.9,
                 text_partial_threshold: float = 0.85,
                 text_model_version: int = 1,
                 use_tf_knn: bool = False,
                 verbose: int = 0) -> None:
        # FIXME: wire indexer params

        super().__init__(batch_size=batch_size,
                         index_type=IndexerType.approx,
                         indexer_params={},
                         store_data=store_data,
                         use_tf_knn=use_tf_knn,
                         text_global_threshold=text_global_threshold,
                         text_partial_threshold=text_partial_threshold,
                         text_model_version=text_model_version,
                         verbose=verbose)
