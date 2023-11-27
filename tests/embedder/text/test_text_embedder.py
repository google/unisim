# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from unisim.embedder.text import TextEmbedder

BATCH_SIZE = 4


def test_basic_properties():
    embedder = TextEmbedder(model_id="text/retsim/v1", batch_size=BATCH_SIZE, verbose=True)
    assert embedder.embedding_size == 256


def test_embed_short_str():
    s = "This is a test. 😀"
    embedder = TextEmbedder(batch_size=BATCH_SIZE)
    embs = embedder.embed([s])
    assert embs[0].shape == (256,)


def test_embed_long_str():
    s = "s" * 1030
    embedder = TextEmbedder(batch_size=BATCH_SIZE)
    embs = embedder.embed([s])
    assert embs[0].shape == (256,)
