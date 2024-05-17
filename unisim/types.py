# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Union

from jaxtyping import Float32
from numpy import ndarray

# Embeddings
Embedding = Float32[ndarray, "embedding"]
BatchEmbeddings = Float32[ndarray, "batch embeddings"]
