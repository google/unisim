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
from functools import cache
from typing import AnyStr, Sequence

import numpy as np

PAD = [0.0] * 24


@cache
def char2bin(chr: str) -> Sequence[float]:
    cp = ord(chr)
    v = format(cp, "#026b")
    return [float(b) for b in v[2:]]


@cache
def token2bin(word: str, add_space: bool = True) -> Sequence[float]:
    "convert a word"
    if add_space:
        word += " "
    return [char2bin(c) for c in word]


def binarize_str(
    txt: AnyStr,
    docid: int,
    chunk_size: int = 512,
    last_chunk_min_size: int = 256,
):
    # binarize
    bins = [char2bin(c) for c in txt]

    # remove last chunk if too small
    lbin = len(bins)
    last_chunk_size = lbin % chunk_size

    # deal with last chunk
    if lbin > chunk_size and last_chunk_size <= last_chunk_min_size:
        # if too small truncate
        if last_chunk_size:  # avoid doing it if by change perfect cut
            bins = bins[: lbin - last_chunk_size]
    else:
        # else pad
        pad_len = chunk_size - len(bins) % chunk_size
        padding = [PAD for _ in range(pad_len)]
        bins.extend(padding)

    # arrayifying and reshaping
    arr = np.array(bins, dtype="float32")
    num_chunks = arr.shape[0] // chunk_size
    arr = np.reshape(arr, (num_chunks, chunk_size, 24))
    return arr, num_chunks, docid


def binarizer(
    txts: Sequence[AnyStr],
    chunk_size: int = 512,
    last_chunk_min_size: int = 256,
):
    inputs = []
    chunk_ids = []
    chunk_ids_cursor = 0
    for docid, txt in enumerate(txts):
        arr, num_chunks, docid = binarize_str(
            txt=txt,
            docid=docid,
            chunk_size=chunk_size,
            last_chunk_min_size=last_chunk_min_size,
        )
        inputs.extend(arr)
        current_chunks_ids = list(range(chunk_ids_cursor, chunk_ids_cursor + num_chunks))
        chunk_ids.append(current_chunks_ids)
        chunk_ids_cursor += num_chunks

    # stack
    inputs = np.stack(inputs, axis=0)
    return inputs, chunk_ids


# def vectorized_binarizer(txts: Sequence[AnyStr], chunk_size: int = 512, last_chunk_min_size: int = 16):
#     inputs = []
#     for docid, txt in enumerate(txts):
#         arr, _, _ = binarize_str(
#             txt=txt, docid=docid, chunk_size=chunk_size, last_chunk_min_size=last_chunk_min_size
#         )  # noqa
#         inputs.append(arr)

#     doc_ids = list(range(len(inputs)))
#     # TODO(ovallis): Convert to dense 3D array and pad with zeros.
#     return inputs, doc_ids


# TODO (marinazh): parallelize the binarizer for efficiency
# def multi_binarizer(txts: Sequence[AnyStr], chunk_size: int = 512):
#     # parallelize
#     promises = []
#     with futures.ProcessPoolExecutor() as executor:
#         for docid, txt in enumerate(txts):
#             promises.append(executor.submit(binarize_str, txt=txt, docid=docid, chunk_size=chunk_size))

#         inputs = []
#         docids = []
#         for promise in futures.as_completed(promises):
#             data = promise.result()
#             arr, num_chunks, docid = data
#             inputs.extend(arr)
#             docids.append([docid] * num_chunks)

#     # stack
#     inputs = np.stack(inputs, axis=0)
#     return inputs, docids
