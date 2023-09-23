from __future__ import annotations
from jaxtyping import Array, Float32, Int64

# any embeddinsg
TensorEmbedding = Float32[Array, 'embedding']
BatchEmbeddings = Float32[Array, 'batch embedding']


# global
GlobalEmbedding = Float32[Array, 'embedding']
BatchGlobalEmbeddings = Float32[Array, 'batch embedding']

# partial embeddings
PartialEmbedding = Float32[Array, 'embedding']
PartialEmbeddings = Float32[Array, 'chunk embedding']
BatchPartialEmbeddings = Float32[Array, 'batch chunk embedding']

# distances
BatchDistances = Float32[Array, 'batch']
BatchDistances2D = Float32[Array, 'batch batch']

# indexes
BatchIndexes = Int64[Array, 'batch']
