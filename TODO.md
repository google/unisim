# TODO

## cleanup backend
Move tfbinarizer to the tf backend
Move binarize to onnx (do probably a dir)
move the TFKNN to backend/tf/dir after refactoring

Don't use np.array or np.asarray - use instead `asanyarray`
see: https://stackoverflow.com/questions/56805126/is-there-a-significant-overhead-in-calling-np-asarray-on-a-numpy-array

## model
- retsim doesn't save as .keras

## speed

- Vectorize the tokenizer
- Check if we want TF or ONNX for single GPU
- Check if we wnat TF or ONNX on singele CPU


- Explore using numba even on CPU to complement onnx (or replace some TF)
- Add multi gpu support
- Check if we want ONNX multigpu or TF multigpu



### tokenizer
- Add remove whitespace / keep stuff()
- add lowercasing
- add tqdm
### to test
- add a cache for words?
- C

### gather
the average vector is done in numpy because we are unable to get TF to do it correctly

### embedeer
check if using TF or np is faster
Add parallel futures with a thread pool potentially

### indexer

Maybe do something smarter like this
https://github.com/unum-cloud/usearch/blob/2b7e32b85090bff6fcc99351d864deefae5f2bec/python/usearch/index.py#L315

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
- Use GPU and TF for exact KNN if exist
- Optimization: Rewrite the code such that we only query the global index
for the elements we have a partial match


## Embeder
- Support fetching models from the internet
- Support in backend detecting onnx cpu vs gpu


## Viz & Clustering
- Support 2d model projection (support viz color)
- Support clustering and integrate the color with Viz (pandas? Arrow?)