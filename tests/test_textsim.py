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
def test_textsim_match(backend_type, index_type):
    set_test_backend(b=backend_type)
    tsim = TextSim(index_type=index_type, model_id="text/retsim/v1", batch_size=BATCH_SIZE)
    queries = ["test", "This is a test", "cookies", "a" * 1024]
    targets = ["this is a test! 😀", "COOKIES", "a" * 1024 + "b" * 256, "test"]

    df = tsim.match(queries=queries, targets=targets)

    expected_df = pd.DataFrame(
        {
            "Query": queries,
            "Match": [targets[3], targets[0], targets[1], targets[2]],
            "Similarity": [1.0000, 0.937122, 0.895018, 1.000000],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)
