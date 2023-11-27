# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import pytest

from unisim.indexer import Indexer

index_type = ["exact", "approx"]
EMBEDDING_SIZE = 3


def set_up_test_indexer(index_type):
    indexer = Indexer(
        embedding_size=EMBEDDING_SIZE,
        index_type=index_type,
        params={},
    )
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")
    idxs = [0, 1]

    indexer.add(embs, idxs)
    return indexer


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_add(index_type):
    indexer = set_up_test_indexer(index_type)
    assert indexer.idxs == [0, 1]

    embs = np.array([[1, 1, 3]], dtype="float32")
    idxs = [2]
    indexer.add(embs, idxs)
    assert indexer.idxs == [0, 1, 2]


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_search_match(index_type):
    indexer = set_up_test_indexer(index_type)

    query_embs = np.array([[1, 1, 1], [0, 0, 0]], dtype="float32")

    result_collection = indexer.search(
        queries=["query1", "query2"],
        query_embeddings=query_embs,
        similarity_threshold=0.5,
        k=2,
        return_data=True,
        data=["test1", "test2"],
    )
    assert result_collection.total_matches == 2

    result = result_collection.results[0]
    assert result.query_idx == 0
    assert result.query_data == "query1"
    assert result.num_matches == 2
    assert len(result.matches) == 2

    match = result.matches[0]
    assert match.idx == 1
    assert match.rank == 0
    assert match.is_match
    assert match.data == "test2"


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_search_non_match(index_type):
    indexer = set_up_test_indexer(index_type)

    query_embs = np.array([[0, 0, 0]], dtype="float32")

    result_collection = indexer.search(
        queries=["query1"],
        query_embeddings=query_embs,
        similarity_threshold=0.5,
        k=1,
        return_data=True,
        data=["test1", "test2"],
    )
    assert result_collection.total_matches == 0

    result = result_collection.results[0]
    assert result.query_idx == 0
    assert result.query_data == "query1"
    assert result.num_matches == 0


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_reset(index_type):
    indexer = set_up_test_indexer(index_type)
    indexer.reset()
    assert indexer.idxs == []

    if indexer.use_exact:
        assert indexer.embeddings == []
    else:
        assert indexer.index.size == 0


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_indexer_save_reload(tmpdir, index_type):
    indexer = set_up_test_indexer(index_type)
    indexer.save(tmpdir)
    indexer.load(tmpdir)

    embs = np.array([[1, 1, 3]], dtype="float32")
    idxs = [2]
    indexer.add(embs, idxs)
    assert indexer.idxs == [0, 1, 2]

    if indexer.use_exact:
        assert len(indexer.embeddings) == 3
    else:
        assert indexer.index.size == 3
