from enum import Enum


class BackendType(Enum):
    unknown = 0
    onnx = 1
    tf = 2


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
