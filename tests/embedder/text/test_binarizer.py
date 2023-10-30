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

from unisim.embedder.text.binarizer import binarizer


def test_binarize():
    s = "This is a test 😀"

    inputs, chunk_ids = binarizer([s])
    assert inputs.shape == (1, 512, 24)
    assert chunk_ids == [[0]]

    inputs, chunk_ids = binarizer([s, s, s])
    assert inputs.shape == (3, 512, 24)
    assert chunk_ids == [[0], [1], [2]]


def test_binarize_long_str():
    s1 = "s" * 1030
    s2 = "This is a test 😀"

    inputs, chunk_ids = binarizer([s1, s2], chunk_size=512, last_chunk_min_size=256)
    assert inputs.shape == (3, 512, 24)
    assert chunk_ids == [[0, 1], [2]]


def test_binarize_lowercase():
    s1 = "THIS IS A TEST 😀"
    s2 = "this is a test 😀"
    inputs, _ = binarizer([s1, s2], lowercase=True)
    assert np.array_equal(inputs[0], inputs[1])


def test_binarize_cleaned_text():
    s1 = "this is a test. 😀"
    s2 = "   this is a \n\t test. 😀   "
    inputs, _ = binarizer([s1, s2], cleanup=True)
    assert np.array_equal(inputs[0], inputs[1])
