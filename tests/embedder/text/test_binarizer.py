# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

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
