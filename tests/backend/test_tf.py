# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import tensorflow as tf
from numpy.testing import assert_almost_equal

from unisim.backend.tf import cosine_similarity, load_model, predict


def test_tf_cosine_similarity():
    emb0 = tf.constant([[0.0, 1.0]])
    emb1 = tf.constant([[1.0, 0.0]])

    cos_sim = cosine_similarity(emb0, emb1)
    assert tuple(tf.shape(cos_sim)) == (1, 1)
    assert tf.reduce_sum(cos_sim) == 0.0

    cos_sim = cosine_similarity(emb0, emb0)
    assert tuple(tf.shape(cos_sim)) == (1, 1)
    assert tf.reduce_sum(cos_sim) == 1.0

    emb = tf.nn.l2_normalize([[0.1, 0.3, 0.2], [0.0, 0.1, 0.5]], axis=-1)
    cos_sim = cosine_similarity(emb, emb)
    assert tuple(tf.shape(cos_sim)) == (2, 2)
    assert_almost_equal(cos_sim[0][0], 1.0, 3)
    assert_almost_equal(cos_sim[0][1], 0.68138516, 3)


def test_tf_load_model(tmp_path):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, activation="relu", input_shape=(4,)))

    save_path = tmp_path / "test_model"
    model.save(save_path)
    load_model(save_path, verbose=True)


def test_tf_predict():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, activation="relu", input_shape=(4,)))

    batch = tf.constant([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    predict(model, batch)
