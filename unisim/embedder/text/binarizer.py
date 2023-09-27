from functools import cache
import numpy as np
from concurrent import futures
from typing import Sequence, AnyStr


PAD = [0.0] * 24


# @cache
# def tobin(codepoint: int) -> Sequence[float]:
#     # 24bits representation
#     # 24 bits + 2byes for 0b at the begining = #026b
#     v = format(codepoint, '#026b')
#     return [float(b) for b in v[2:]]


@cache
def char2bin(chr: str) -> Sequence[float]:
    cp = ord(chr)
    v = format(cp, '#026b')
    return [float(b) for b in v[2:]]


@cache
def token2bin(word: str) -> Sequence[float]:
    "convert a word "
    word += " "
    return [char2bin(c) for c in word]


def binarize_str(txt: AnyStr, docid: int, chunk_size: int = 512,
                 cleanup: bool = True, lowercase: bool = False,
                 last_chunk_min_size: int = 16):

    if lowercase:
        txt = txt.lower()

    # binarize
    if cleanup:
        # note: we use token2bin because we hope to leverage caching per token
        words = txt.split()
        bins = []
        for w in words:
            bins.extend(token2bin(w))
    else:
        bins = [char2bin(c) for c in txt]

    # remove last chunk if too small
    lbin = len(bins)
    last_chunk_size = lbin % chunk_size

    # deal with last chunk
    if lbin > chunk_size and last_chunk_size <= last_chunk_min_size:
        # if too small truncate
        if last_chunk_size:  # avoid doing it if by change perfect cut
            bins = bins[:lbin-last_chunk_size]
    else:
        # else pad
        pad_len = chunk_size - len(bins) % chunk_size
        padding = [PAD for _ in range(pad_len)]
        bins.extend(padding)

    # arrayifying and reshaping
    arr = np.array(bins, dtype='float32')
    num_chunks = arr.shape[0] // chunk_size
    arr = np.reshape(arr, (num_chunks, chunk_size, 24))
    return arr, num_chunks, docid


def binarizer(txts: Sequence[AnyStr],
              chunk_size: int = 512,
              last_chunk_min_size: int = 16):

    inputs = []
    chunk_ids = []
    chunk_ids_cursor = 0
    for docid, txt in enumerate(txts):
        arr, num_chunks, docid = binarize_str(txt=txt,
                                              docid=docid,
                                              chunk_size=chunk_size,
                                              last_chunk_min_size=last_chunk_min_size)  # noqa
        inputs.extend(arr)
        current_chunks_ids = np.arange(chunk_ids_cursor,
                                       chunk_ids_cursor + num_chunks)
        chunk_ids.append(current_chunks_ids)
        chunk_ids_cursor += num_chunks

    # stack
    inputs = np.stack(inputs, axis=0)
    return inputs, chunk_ids


def multi_binarizer(txts: Sequence[AnyStr], chunk_size: int = 512):

    # parallelize
    promises = []
    with futures.ProcessPoolExecutor() as executor:
        for docid, txt in enumerate(txts):
            promises.append(executor.submit(binarize_str, txt=txt, docid=docid,
                                            chunk_size=chunk_size))
        # collect
        inputs = []
        docids = []
        for promise in futures.as_completed(promises):
            data = promise.result()
            arr, num_chunks, docid = data
            inputs.extend(arr)
            docids.append([docid] * num_chunks)

    # stack
    inputs = np.stack(inputs, axis=0)
    return inputs, docids
