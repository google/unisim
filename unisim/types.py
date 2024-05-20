# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import TYPE_CHECKING, Union

from jaxtyping import Float32

if TYPE_CHECKING:
    from numpy import ndarray
    from tensorflow import Tensor

    Array = Union[Tensor, ndarray]
else:
    from numpy import ndarray as Array

# Embeddings
Embedding = Float32[Array, "embedding"]
BatchEmbeddings = Float32[Array, "batch embeddings"]
