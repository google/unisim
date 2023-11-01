"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
from __future__ import annotations

from typing import Union

from jaxtyping import Float32, Int64
from numpy import ndarray
from tensorflow import Tensor

Array = Union[Tensor, ndarray]

# any embeddinsg
TensorEmbedding = Float32[Array, "embedding"]
BatchEmbeddings = Float32[Array, "batch embedding"]

# global
GlobalEmbedding = Float32[Array, "embedding"]
BatchGlobalEmbeddings = Float32[Array, "batch embedding"]

# partial embeddings
PartialEmbedding = Float32[Array, "embedding"]
PartialEmbeddings = Float32[Array, "chunk embedding"]
BatchPartialEmbeddings = Float32[Array, "batch chunk embedding"]

# distances
BatchDistances = Float32[Array, "batch"]
BatchDistances2D = Float32[Array, "batch batch"]

# indexes
BatchIndexes = Int64[Array, "batch"]
