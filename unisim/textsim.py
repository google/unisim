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

from tabulate import tabulate

from .dataclass import Result
from .embedder import TextEmbedder
from .enums import IndexerType
from .types import BatchGlobalEmbeddings, BatchPartialEmbeddings
from .unisim import UniSim


class TextSim(UniSim):
    def __init__(
        self,
        store_data: bool = True,
        global_threshold: float = 0.9,
        partial_threshold: float = 0.85,
        index_type: str | IndexerType = "exact",
        batch_size: int = 128,
        use_accelerator: bool = True,
        model_id: str = "text/retsim/v1",
        index_params: Dict[str, Any] | None = None,
        verbose: int = 0,
    ) -> None:
        embedder = TextEmbedder(
            batch_size=batch_size,
            model_id=model_id,
            verbose=verbose,
        )
        super().__init__(
            store_data=store_data,
            global_threshold=global_threshold,
            partial_threshold=partial_threshold,
            index_type=index_type,
            batch_size=batch_size,
            use_accelerator=use_accelerator,
            model_id=model_id,
            embedder=embedder,
            index_params=index_params,
            verbose=verbose,
        )

    def similarity(self, input_1: str, input_2: str) -> float:
        return super().similarity(input_1, input_2)

    def embed(self, inputs: Sequence[str]) -> Tuple[BatchGlobalEmbeddings, BatchPartialEmbeddings]:
        return super().embed(inputs)

    def add(self, inputs: Sequence[str]) -> Sequence[int]:
        return super().add(inputs)

    def search(self, inputs: Sequence[str], gk: int = 5, pk: int = 5):
        return super().search(inputs, gk, pk)

    def visualize(self, result: Result, max_display_chars: int = 32):
        rows = []
        for m in result.matches:
            # check content
            gs = round(m.global_similarity, 2) if m.global_similarity else ""
            ps = round(m.partial_similarity, 2) if m.partial_similarity else ""
            content = m.data[:max_display_chars] if m.data else ""

            # build row
            row = [m.idx, m.is_global_match, gs, m.is_partial_match, ps, content]
            rows.append(row)
        if result.query:
            print(f'Query {result.query_idx}: "{result.query}"')
        else:
            print(f"Query {result.query_idx}")
        print(tabulate(rows, headers=["idx", "is_global", "global_sim", "is_partial", "partial_sim", "content"]))
