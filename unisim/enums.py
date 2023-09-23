from enum import Enum


class AcceleratorType(Enum):
    unknown = 0
    cpu = 1
    gpu = 2
    tpu = 3


class IndexerType(Enum):
    exact = 0
    approx = 1


class ModalityType(Enum):
    multimodal = 0
    text = 1
    image = 2
