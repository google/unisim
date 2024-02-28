# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from importlib import reload

import pandas as pd
import pytest

import unisim
from unisim import TextSim
from unisim.config import BackendType, set_backend

backend_type = ["tf", "onnx"]
index_type = ["exact", "approx"]
BATCH_SIZE = 4


def set_test_backend(b):
    # Reload unisim to pick up the new backend
    os.environ["BACKEND"] = b

    if b == "tf":
        set_backend(BackendType.tf)
    elif b == "onnx":
        set_backend(BackendType.onnx)

    reload(unisim)


@pytest.mark.parametrize("backend_type", backend_type, ids=backend_type)
@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_textsim_similarity(backend_type, index_type):
    set_test_backend(b=backend_type)
    tsim = TextSim(index_type=index_type, model_id="text/retsim/v1", batch_size=BATCH_SIZE, verbose=True)

    sim = tsim.similarity("I love icecreams", "I love icecreams")
    assert round(sim, 3) == 1

    test_str = "b" * 512 + "a" * 512
    sim = tsim.similarity(test_str, test_str)
    assert round(sim, 3) == 1

    sim = tsim.similarity("doubting dreams", "rough winds do shake the darling buds of may,")
    assert round(sim, 3) == 0.485

    sim = tsim.similarity("this is a test", "This is a test 😀")
    assert round(sim, 3) == 0.967


@pytest.mark.parametrize("backend_type", backend_type, ids=backend_type)
@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_textsim_embed(backend_type, index_type):
    set_test_backend(b=backend_type)
    tsim = TextSim(index_type=index_type, model_id="text/retsim/v1", batch_size=BATCH_SIZE)

    embs = tsim.embed(["This is a test"])
    assert embs.shape == (1, 256)

    embs = tsim.embed(["a" * 2000, "test"])
    assert embs.shape == (2, 256)


@pytest.mark.parametrize("backend_type", backend_type, ids=backend_type)
@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_textsim_add(backend_type, index_type):
    set_test_backend(b=backend_type)
    tsim = TextSim(index_type=index_type, model_id="text/retsim/v1", batch_size=BATCH_SIZE)

    s1 = "This is a test"
    s2 = "a" * 2000

    tsim.add([s1])
    tsim.add([s1, s2, s2])


@pytest.mark.parametrize("backend_type", backend_type, ids=backend_type)
@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_textsim_basic_search_workflow(backend_type, index_type):
    set_test_backend(b=backend_type)
    tsim = TextSim(index_type=index_type, model_id="text/retsim/v1", batch_size=BATCH_SIZE)
    tsim.info()
    tsim.reset_index()

    s1 = "This is a test"
    s2 = "a" * 2000
    s3 = "cookies"
    s4 = "cookies?"
    tsim.add([s1, s2, s3, s4])

    rc = tsim.search([s3], k=1)
    res_0 = rc.results[0]
    tsim.visualize(res_0)

    assert rc.total_matches == 1
    assert res_0.num_matches == 1

    rc = tsim.search([s3, s1], k=5)
    res_0 = rc.results[0]
    res_1 = rc.results[1]
    tsim.visualize(res_0)

    assert rc.total_matches == 3
    assert res_0.num_matches == 2
    assert res_0.query_idx == 0
    assert res_0.query_data == s3
    assert res_0.query_embedding.shape == (256,)

    assert res_1.num_matches == 1
    assert res_0.matches[0].idx == 2
    assert res_0.matches[1].idx == 3
    assert res_1.matches[0].idx == 0
    assert res_1.matches[0].embedding.shape == (256,)


@pytest.mark.parametrize("backend_type", backend_type, ids=backend_type)
@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_textsim_match_two_lists(backend_type, index_type):
    set_test_backend(b=backend_type)
    tsim = TextSim(index_type=index_type, model_id="text/retsim/v1", batch_size=BATCH_SIZE)
    queries = ["test", "This is a test", "cookies", "a" * 1024]
    targets = ["this is a test! 😀", "COOKIES", "a" * 1024 + "b" * 256, "test"]

    df = tsim.match(queries=queries, targets=targets)

    expected_df = pd.DataFrame(
        {
            "query": queries,
            "target": [targets[3], targets[0], targets[1], targets[2]],
            "similarity": [1.0000, 0.9371, 0.8950, 1.000],
            "is_match": [True, True, False, True],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df, check_exact=False, atol=1e-4)


@pytest.mark.parametrize("backend_type", backend_type, ids=backend_type)
@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_textsim_match_single_list(backend_type, index_type):
    set_test_backend(b=backend_type)
    tsim = TextSim(index_type=index_type, model_id="text/retsim/v1", batch_size=BATCH_SIZE)
    queries = ["test", "This is a test", "this is a test! 😀", "a" * 1024]

    df = tsim.match(queries)

    expected_df = pd.DataFrame(
        {
            "query": queries,
            "target": [queries[1], queries[2], queries[1], queries[0]],
            "similarity": [0.7674, 0.9371, 0.9371, 0.3772],
            "is_match": [False, True, True, False],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df, check_exact=False, atol=1e-4)
