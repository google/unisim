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
from unisim.embedder.text import TextEmbedder

BATCH_SIZE = 4


def test_basic_properties():
    embedder = TextEmbedder(model_id="text/retsim/v1", batch_size=BATCH_SIZE, verbose=True)

    assert embedder.embedding_size == 256
    assert embedder.chunk_size == 512


def test_embed_short_str():
    s = "This is a test. 😀"
    embedder = TextEmbedder(batch_size=BATCH_SIZE)

    global_embs, partial_embs = embedder.embed([s])

    assert global_embs[0].shape == (256,)
    assert partial_embs[0].shape == (1, 256)


def test_embed_long_str():
    s = "s" * 1030
    embedder = TextEmbedder(batch_size=BATCH_SIZE)

    global_embs, partial_embs = embedder.embed([s])

    assert global_embs[0].shape == (256,)
    assert partial_embs[0].shape == (2, 256)
