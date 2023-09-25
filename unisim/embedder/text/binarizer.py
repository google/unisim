from functools import cache
import numpy as np
from concurrent import futures
from typing import Sequence, AnyStr


PAD = [0.0] * 24


@cache
def tobin(codepoint: int) -> Sequence[float]:
    # 24bits representation
    # 24 bits + 2byes for 0b at the begining = #026b
    v = format(codepoint, '#026b')
    return [float(b) for b in v[2:]]


@cache
def char2bin(chr) -> Sequence[float]:
    cp = ord(chr)
    v = format(cp, '#026b')
    return [float(b) for b in v[2:]]


def binarize_str(txt: AnyStr, docid: int, chunk_size: int = 512):
    if isinstance(txt, str):
        mstr = txt
    else:
        mstr = str(txt)

    # binarize
    bins = [char2bin(c) for c in mstr]

    # padding
    pad_len = chunk_size - len(bins) % chunk_size
    padding = [PAD for _ in range(pad_len)]
    bins.extend(padding)

    # arrayifying and reshaping
    arr = np.array(bins, dtype='float32')
    num_chunks = arr.shape[0] // chunk_size
    arr = np.reshape(arr, (num_chunks, chunk_size, 24))
    return arr, num_chunks, docid


def binarizer(txts: Sequence[AnyStr], chunk_size: int = 512):

    inputs = []
    docids = []
    for docid, txt in enumerate(txts):
        arr, num_chunks, docid = binarize_str(txt=txt,
                                              docid=docid,
                                              chunk_size=chunk_size)
        inputs.extend(arr)
        docids.append([docid] * num_chunks)

    # stack
    inputs = np.stack(inputs, axis=0)
    return inputs, docids


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
