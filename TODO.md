# TODO

- Fix



## model
- retsim doesn't save as .keras

## speed

- Explore using numba even on CPU to complement onnx (or replace some TF)

### tokenizer
- try to cache the np.array in tobin() see if we can make the tokenizer even faster - (alternative C)
- do thread computation
- add tqdm
- add a cache for words?
- support lowercasing
- C

### gather
the average vector is done in numpy because we are unable to get TF to do it correctly

### embedeer
try to check if using REALNP or NP is better for averaging the chunk in text_embedder
than using TF

### indexer

- Add representation to dataclass
 - partialmatch
 - similarity


## colab

add benchmark colab

## Unisim
- Support ANN and KNN

## Match
- Review output and fix

## Text
- check if using numpy for the reduce average is fast enough. Best guess
you don't want to move such small computation to GPU.

## Benchmark
- Write a small benchmark to test various speed improvement

## Indexer
- Use GPU via a small onnx model for exact KNN
- Add ANN support

## Embeder
- Support fetching models from the internet
- Support in backend detecting onnx cpu vs gpu


## Viz & Clustering
- Support 2d model projection (support viz color)
- Support clustering and integrate the color with Viz (pandas? Arrow?)