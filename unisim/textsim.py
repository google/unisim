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
    """TextSim class for computing textual similarity.

    TextSim leverages the RETSim embedding model to compute the syntactic
    similarity between texts, retrieve similar documents, and deduplicate
    datasets. The model is a highly-optimized, small model that runs
    efficiently on both GPU and CPU, which makes TextSim significantly faster
    than traditional edit distance-based techniques and other neural-based
    embeddings.

    For more details on the RETSim model, please see TODO (marinazh) [paper link].
    """

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
        """Create TextSim to index, search, and compute similarity between
        texts using RETSim embedding model.

        Args:
            store_data: Whether to store a copy of the indexed data and return
                data when returning search results. For large datasets
                (e.g. millions of items), we recommend setting store_data=False
                to reduce memory usage.

            index_type: Indexer type, either "exact" for exact search or
                "approx" for Approximate Nearest Neighbor (ANN) search. Using
                exact search guarantees you will return the most similar texts
                to your search queries every single time. We recommend using
                "approx" for large datasets for computational efficiency and it
                is accurate >99% of the time.

            return_embeddings: Whether to return the embeddings corresponding
                to each search result when calling TextSim.search().

            batch_size: Batch size for model inference.

            use_accelerator: Whether to use an accelerator (GPU), if available.

            model_id: String id of the model. Defaults to "text/retsim/v1" for
                the RETSim model.

            index_params: Additional parameters to be passed into the Indexer,
                only used when index_type is "approx" (ANN search). Supported
                dict keys include {dtype, connectivity, expansion_add,
                expansion_search}, please see the USearch Index documentation
                for more details (https://github.com/unum-cloud/usearch).

            verbose: Verbosity level, set to 1 for verbose.
        """
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
        """Return the similarity between two texts as a float.

        Args:
            input1: First input.

            input2: Second input.

        Returns:
            Similarity between the two texts, a value between 0 and 1.
        """
        return super().similarity(input1=input1, input2=input2)

    def match(self, queries: Sequence[Any], targets: Sequence[Any]):
        """Find the closest matches for queries in a list of targets and
        return the result as a pandas DataFrame.

        Args:
            queries: Input query texts to search for.

            targets: Target texts to search for queries in. For each query,
                we search and return the most similar text in `targets`.

        Returns:
            Returns a pandas DataFrame with ["Query", "Match", "Similarity"]
            columns, representing each query, nearest match in `targets`, and
            their similarity value.
        """
        return super().match(queries=queries, targets=targets)

    def embed(self, inputs: Sequence[str]) -> BatchEmbeddings:
        """Compute embeddings for input texts.

        Args:
            inputs: Sequence of input texts to embed.

        Returns:
            Embeddings for each text in `inputs`. The RETSim model embeds
            each input text into a 256-dimensional float32 array.
        """
        return super().embed(inputs=inputs)

    def add(self, inputs: Sequence[str]) -> List[int]:
        """Embed and add inputs to the index.

        Args:
            inputs: Inputs to embed and add to the index.

        Returns:
             idxs: Indices corresponding to the inputs added to the index,
             where idxs[0] is the idx for inputs[0] in the index.
        """
        return super().add(inputs=inputs)

    def search(self, inputs: Sequence[str], similarity_threshold: float = 0.9, k: int = 1) -> ResultCollection:
        """Search for and return the k closest matches for a set of queries,
        and mark the ones that are closer than `similarity_threshold` as
        near-duplicate matches.

        Args:
            inputs: Query input texts to search for.

            similarity_threshold: Similarity threshold for near-duplicate
                match, where a query and a search result are considered
                near-duplicate matches if their similarity is higher than
                `similarity_threshold`. 0.9 is typically a good threshold
                for detecting near-duplicate matches.

            k: Number of nearest neighbors to lookup for each query input.

        Returns
            result_collection: ResultCollection containing the search results.
            The total number of near-duplicate matches can be accessed through
            result_collection.total_matches and the list of results for each
            query can be accessed through result_collection.results.
        """
        return super().search(inputs=inputs, similarity_threshold=similarity_threshold, k=k)

    def visualize(self, result: Result, max_display_chars: int = 32):
        """Visualize a search result, displaying the query, closest
        search results and matches, and their similarity values.

        Args:
            result: Result to visualize.

            max_display_chars: Max number of characters to display for text.
        """
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
