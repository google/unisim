from functools import cache
import numpy as np
from typing import Sequence, AnyStr

PAD = [0.0] * 24


@cache
def tobin(codepoint: int) -> Sequence[float]:
    # 24bits representation
    # 24 bits + 2byes for 0b at the begining = #026b
    v = format(codepoint, '#026b')
    return [float(b) for b in v[2:]]


def binarizer(txts: Sequence[AnyStr],
              chunk_size: int = 512):

    inputs = []
    docids = []
    # FIXME parallelize
    for docid, txt in enumerate(txts):
        # ord() doesn't supports bytes
        if isinstance(txt, str):
            mstr = txt
        else:
            mstr = str(txt)

        # to utf8 codepoints
        codepoints = [ord(c) for c in mstr]

        # binarize
        bins = [tobin(c) for c in codepoints]

        # padding
        pad_len = chunk_size - len(bins) % chunk_size
        padding = [PAD for _ in range(pad_len)]
        bins.extend(padding)

        # arrayifying and reshaping
        arr = np.array(bins, dtype='float32')
        num_chunks = arr.shape[0] // chunk_size
        arr = np.reshape(arr, (num_chunks, chunk_size, 24))

        # collect
        inputs.extend(arr)
        docids.append([docid] * num_chunks)

    # stack
    inputs = np.stack(inputs, axis=0)
    return inputs, docids
