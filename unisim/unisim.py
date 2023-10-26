from .config import get_accelerator
from .embedder import TextEmbedder
from .enums import AcceleratorType, IndexerType
from .modality import Modality
from .viz import Visualization


class TextSim:
    def __init__(
        self,
        store_data: bool = True,
        global_threshold: float = 0.9,
        partial_threshold: float = 0.85,
        index_type: str | IndexerType = "exact",
        batch_size: int = 128,
        use_accelerator: bool = True,
        model_id: str = "text/retsim/1",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.store_data = store_data
        self.global_threshold = global_threshold
        self.partial_threshold = partial_threshold
        self.index_type = index_type if isinstance(index_type, IndexerType) else IndexerType[index_type]
        self.batch_size = batch_size
        self.use_accelerator = use_accelerator
        self.verbose = verbose

        if self.store_data:
            print("UniSim is storing a copy of the indexed data")
            print("if you are using large data corpus consider disable this behavior using store_data=False")
        else:
            print("UniSim is not storing a copy of the index data to save memory")
            print("If you want to store it use store_data=True")

        if use_accelerator and not get_accelerator() == AcceleratorType.gpu:
            print("Accelerator is not available, using cpu instead")
            self.use_accelerator = False

        # viz
        self.viz = Visualization()

        self.text_embedder = TextEmbedder(
            batch_size=self.batch_size,
            version=self.model_id,
            verbose=self.verbose,
        )

        # Initalizing the models/embedders
        self.modality = Modality(
            store_data=self.store_data,
            global_threshold=self.global_threshold,
            partial_threshold=self.partial_threshold,
            index_type=self.index_type,
            batch_size=self.batch_size,
            embedding_model=self.text_embedder,
            model_id=self.model_id,
            index_params=None,
            verbose=self.verbose,
        )

    def info(self):
        print("[Embedder]")
        print(f"|-batch_size: {self.batch_size}")
        print("[Indexer]")
        print(f"|-index_type: {self.index_type.name}")
        print(f"|-use_accelerator: {self.use_accelerator}")
        print(f"|-store index data: {self.store_data}")

    def similarity(self, input_1: str, input_2: str) -> float:
        return self.modality.similarity(input_1, input_2)
