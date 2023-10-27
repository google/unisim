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
from typing import Any, Dict, Sequence, Tuple
import numpy as np


def flatten_partial_embeddings(
        batch_partial_embeddings,
        ges_idxs: Sequence[int] | None = None,
    ):
        "Flatten partial embeddings and remap idxs to global ones"
        if ges_idxs is None:
            ges_idxs = []
        flatten_pes = []
        pes_idxs = []
        for idx, pes in enumerate(batch_partial_embeddings):
            for pe in pes:
                flatten_pes.append(pe)
                if ges_idxs:
                    pes_idxs.append(ges_idxs[idx])
        return np.asanyarray(flatten_pes), pes_idxs


def get_batched_inputs(inputs: Sequence[Any], batch_size: int):
    for b_offset in range(0, len(inputs), batch_size):
        batch = inputs[b_offset : b_offset + batch_size]
        yield batch
