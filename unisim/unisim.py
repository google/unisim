from abc import ABC
from .enums import IndexerType, ModalityType
from .modality import Modality


class UniSim(ABC):

    text: Modality
    # FIXME: add image when ready
    image: Modality
    multimodal: Modality

    def __init__(self,
                 index_type: IndexerType,
                 text_global_threshold: float = 0.9,
                 text_partial_threshold: float = 0.9,
                 text_model_version: int = 1,
                 verbose: int = 0) -> None:

        super().__init__()
        self.verbose = verbose
        self.index_type = index_type

        # Initalizing the models/embedders
        self.text = Modality(global_threshold=text_global_threshold,
                             partial_threshold=text_partial_threshold,
                             modality=ModalityType.text,
                             model_version=text_model_version,
                             verbose=verbose)


class ExactUniSim(UniSim):
    """Instanciate UniSim to perform exact matching

    Exact matching guarantee that you get exactly the right results at the cost
    of requiring N^2 computation. Works well in particular
    with a GPU sub 1M points.

    """
    def __init__(self,
                 text_global_threshold: float = 0.9,
                 text_partial_threshold: float = 0.9,
                 text_model_version: int = 1,
                 verbose: int = 0) -> None:

        super().__init__(IndexerType.approx,
                         text_global_threshold=text_global_threshold,
                         text_partial_threshold=text_partial_threshold,
                         text_model_version=text_model_version,
                         verbose=verbose)
