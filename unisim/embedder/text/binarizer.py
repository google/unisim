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
from typing import List, Sequence, Tuple

import numpy as np

PAD = [0.0] * 24


@cache
def char2bin(chr: str) -> List[float]:
    cp = ord(chr)
    v = format(cp, "#026b")
    return [float(b) for b in v[2:]]


@cache
def token2bin(word: str, add_space: bool = True) -> List[List[float]]:
    "convert a word"
    if add_space:
        word += " "
    return [char2bin(c) for c in word]


def binarize_str(
    txt: str,
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
    txts: Sequence[str],
    chunk_size: int = 512,
    last_chunk_min_size: int = 256,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Binarize text using the RETVec chararcter encoder, as detailed in https://arxiv.org/abs/2302.09207.

    Args:
        txts: Input texts to binarize.

        chunk_size: Chunk size in characters to break up the text.

        last_chunk_min_size: Drop last chunk if it has less than `last_chunk_min_size` characters.

    Returns:
        Binarized text which is ready to be embedded using the RETSim embedding model.
    """
    inputs = []
    chunk_ids = []
    chunk_ids_cursor = 0
    for docid, txt in enumerate(txts):
        arr, num_chunks, docid = binarize_str(
            txt=txt, docid=docid, chunk_size=chunk_size, last_chunk_min_size=last_chunk_min_size
        )
        inputs.extend(arr)
        current_chunks_ids = list(range(chunk_ids_cursor, chunk_ids_cursor + num_chunks))
        chunk_ids.append(current_chunks_ids)
        chunk_ids_cursor += num_chunks

    # stack
    binarized_inputs: np.ndarray = np.stack(inputs, axis=0)
    return binarized_inputs, chunk_ids
