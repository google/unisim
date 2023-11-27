# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from enum import Enum


class BackendType(Enum):
    """Backend type.

    One of {unknown, onnx, tf}.
    """

    unknown = 0
    onnx = 1
    tf = 2


class AcceleratorType(Enum):
    """Accelerator type

    One of {unknown, cpu, gpu}.
    """

    unknown = 0
    cpu = 1
    gpu = 2


class IndexerType(Enum):
    """Indexer type

    `exact` for exact search index, `approx` for Approximate Nearest Neighbor (ANN) search index.
    """

    exact = 0
    approx = 1


class ModalityType(Enum):
    """Modality type

    One of {multimodal, text, image}.

    NOTE: Only text is supported in the initial v0 release using unisim.TextSim.
    """

    multimodal = 0
    text = 1
    image = 2
