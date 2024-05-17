# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
from numpy.testing import assert_almost_equal

from unisim.backend import cosine_similarity


def test_onnx_cosine_similarity():
    emb0 = np.array([[0.0, 1.0]])
    emb1 = np.array([[1.0, 0.0]])

    cos_sim = cosine_similarity(emb0, emb1)
    assert tuple(cos_sim.shape) == (1, 1)
    assert np.sum(cos_sim) == 0.0

    cos_sim = cosine_similarity(emb0, emb0)
    assert tuple(cos_sim.shape) == (1, 1)
    assert np.sum(cos_sim) == 1.0

    emb = np.array([[0.1, 0.3, 0.2], [0.0, 0.1, 0.5]])
    emb = emb / np.expand_dims(np.linalg.norm(emb, axis=1), 1)
    cos_sim = cosine_similarity(emb, emb)

    assert tuple(cos_sim.shape) == (2, 2)
    assert_almost_equal(cos_sim[0][0], 1.0, 3)
    assert_almost_equal(cos_sim[0][1], 0.68138516, 3)
