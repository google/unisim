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
from typing import Any, Dict, List, Sequence

from tabulate import tabulate

from .dataclass import Result, ResultCollection
from .embedder import TextEmbedder
from .enums import IndexerType
from .types import BatchEmbeddings
from .unisim import UniSim


class TextSim(UniSim):
    def __init__(
        self,
        store_data: bool = True,
        index_type: str | IndexerType = "exact",
        return_embeddings: bool = True,
        batch_size: int = 128,
        use_accelerator: bool = True,
        model_id: str = "text/retsim/v1",
        index_params: Dict[str, Any] | None = None,
        verbose: int = 0,
    ):
        embedder = TextEmbedder(
            batch_size=batch_size,
            model_id=model_id,
            verbose=verbose,
        )
        super().__init__(
            store_data=store_data,
            index_type=index_type,
            return_embeddings=return_embeddings,
            batch_size=batch_size,
            use_accelerator=use_accelerator,
            model_id=model_id,
            embedder=embedder,
            index_params=index_params,
            verbose=verbose,
        )

    def similarity(self, input1: str, input2: str) -> float:
        return super().similarity(input1=input1, input2=input2)

    def match(self, queries: Sequence[Any], targets: Sequence[Any]):
        return super().match(queries=queries, targets=targets)

    def embed(self, inputs: Sequence[str]) -> BatchEmbeddings:
        return super().embed(inputs=inputs)

    def add(self, inputs: Sequence[str]) -> List[int]:
        return super().add(inputs=inputs)

    def search(self, inputs: Sequence[str], similarity_threshold: float = 0.9, k: int = 1) -> ResultCollection:
        return super().search(inputs=inputs, similarity_threshold=similarity_threshold, k=k)

    def visualize(self, result: Result, max_display_chars: int = 32):
        rows = []
        for m in result.matches:
            # check content
            sim = round(m.similarity, 2) if m.similarity else ""
            content = m.data[:max_display_chars] if m.data else ""

            # build row
            row = [m.idx, m.is_match, sim, content]
            rows.append(row)
        if result.query_data:
            print(f'Query {result.query_idx}: "{result.query_data}"')
        else:
            print(f"Query {result.query_idx}")
        print(tabulate(rows, headers=["idx", "is_match", "similarity", "content"]))
