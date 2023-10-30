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
import numpy as np
import pytest

from unisim.indexer import Indexer

index_type = ["exact", "approx"]
EMBEDDING_SIZE = 3
GLOBAL_THRESHOLD = 0.5
PARTIAL_THREHOLD = 0.5


def set_up_test_indexer(index_type):
    indexer = Indexer(
        embedding_size=EMBEDDING_SIZE,
        global_threshold=GLOBAL_THRESHOLD,
        partial_threshold=PARTIAL_THREHOLD,
        index_type=index_type,
        params={}
    )
    global_embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")
    global_idxs = [0, 1]
    partial_embs = np.array([[1, 1, 3], [3, 1, 2], [1, 2, 3]], dtype="float32")
    partial_idxs = [0, 1, 1]

    indexer.add(global_embs, global_idxs, partial_embs, partial_idxs)
    return indexer


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_add(index_type):
    indexer = set_up_test_indexer(index_type)

    assert indexer.usearch_pkeys_count == 3
    assert indexer.global_idxs == [0, 1]
    assert indexer.partial_idx_2_global_idx == [0, 1, 1]

    global_embs = np.array([[1, 1, 3]], dtype="float32")
    global_idxs = [2]
    partial_embs = np.array([[1, 1, 3], [3, 1, 2], [1, 2, 3]], dtype="float32")
    partial_idxs = [2, 2, 2]

    indexer.add(global_embs, global_idxs, partial_embs, partial_idxs)
    assert indexer.usearch_pkeys_count == 6
    assert indexer.global_idxs == [0, 1, 2]
    assert indexer.partial_idx_2_global_idx == [0, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_search_match(index_type):
    indexer = set_up_test_indexer(index_type)

    global_query_emb = np.array([[1, 1, 1], [0, 0, 0]], dtype="float32")
    global_partial_emb = np.array([[1, 1, 1], [0, 0, 0]], dtype="float32")

    result_collection = indexer.search(
        global_query_emb,
        global_partial_emb,
        gk=2,
        pk=2,
        return_data=True,
        queries=["query1", "query2"],
        data=["test1", "test2"],
    )
    assert result_collection.total_global_matches == 2
    assert result_collection.total_partial_matches == 2

    result = result_collection.results[0]
    assert result.query_idx == 0
    assert result.query == "query1"
    assert result.num_global_matches == 2
    assert result.num_partial_matches == 2
    assert len(result.matches) == 2

    match = result.matches[0]
    assert match.idx == 1
    assert match.global_rank == 0
    assert match.partial_rank == 1
    assert match.is_global_match
    assert match.is_partial_match
    assert match.data == "test2"


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_search_non_match(index_type):
    indexer = set_up_test_indexer(index_type)

    global_query_emb = np.array([[0, 0, 0]], dtype="float32")
    global_partial_emb = np.array([[0, 0, 0]], dtype="float32")

    result_collection = indexer.search(
        global_query_emb,
        global_partial_emb,
        gk=1,
        pk=1,
        return_data=True,
        queries=["query1"],
        data=["test1", "test2"],
    )
    assert result_collection.total_global_matches == 0
    assert result_collection.total_partial_matches == 0

    result = result_collection.results[0]
    assert result.query_idx == 0
    assert result.query == "query1"
    assert result.num_global_matches == 0
    assert result.num_partial_matches == 0


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_reset(index_type):
    indexer = set_up_test_indexer(index_type)
    indexer.reset()
    assert indexer.global_idxs == []
    assert indexer.partial_idx_2_global_idx == []
    assert indexer.usearch_pkeys_count == 0
