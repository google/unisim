# UniSim: Universal Similarity

UniSim is a package for efficiently computing similarity, performing fuzzy matching, deduplicating datasets, and clustering data (text and images). The UniSim package is in beta and we may push breaking changes. The UniSim package currently supports text (e.g. for fuzzy string matching) and image support will be added soon.

## Installation
You can use `pip` to install the latest version of UniSim:

```
pip install unisim
```

By default, UniSim uses [Onnx](https://github.com/onnx/onnx) when running on CPU, and [TensorFlow](https://www.tensorflow.org/) for GPU acceleration. If you have a GPU, you can additionally install TensorFlow using:

```
pip install unisim[tensorflow]
```

## Text UniSim (TextSim)

The goal of TextSim is to provide an easy-to-use tool for efficient, accurate and multilingual fuzzy string matching, near-duplicate detection, and string similarity. Please see the tutorial [colab](notebooks/unisim_text_demo.ipynb) for an in-depth example on using TextSim for real-world use cases like fuzzy matching for addresses.

TextSim is significantly faster than edit-distance algorithms such as Levenshtein Distance for fuzzy string matching and more accurate than ngram-based methods such as MinHash for near-duplicate text detection and clustering. TextSim accepts strings of arbitrary length and can scale to datasets with millions of examples.

To accomplish this, TextSim leverages the RETSim[TODO add link] model to efficiently embed texts into high-dimensional vectors that can be compared using cosine similarity. TextSim then uses [USearch](https://github.com/unum-cloud/usearch) for fast vector search.

### Basic Usage

You can compute the similarity between two strings using the `.similarity(text1, text2)` function. The similarity is a float between 0 and 1, with 1.0 representing most similar (identical strings). The similarity value is the cosine similarity between the vector representations of strings. You can directly get the vector representation of strings using the `.embed(inputs)` function as well.

```python
from unisim import TextSim

text_sim = TextSim()

# compute similarity between two strings
text_sim.similarity("this is a text", "This is a txt! 😀")  # 0.9113
text_sim.similarity("this is a text", "apples")  # 0.4220
```

TextSim offers efficient fuzzy string matching between two lists using the `.match` function. The `.match` function accepts `queries` (list of strings you want to find matches for) and `targets` (list of strings you are finding matches in).

`.match(queries, targets)` returns a Pandas DataFrame, where each row contains a query, its most similar match found in targets, their similarity, and whether or not they are a match (if their similarity is >= `similarity_threshold`). By default, `0.9` is typically a good `similarity_threshold` for near-duplicate matching strings.

```python
from unisim import TextSim

text_sim = TextSim()

queries = ["apple", "appl", "icecream", "house", "random"]
targets = ["apple", "ice cream", "mouse"]

results_df = text_sim.match(queries, targets, similarity_threshold=0.9)
```

This gives you the following Pandas DataFrame of (fuzzy) matches:
```
      query     target  similarity  is_match
0     apple      apple    1.000000      True
1      appl      apple    0.914230      True
2  icecream  ice cream    0.950734      True
3     house      mouse    0.760066     False
4    random      mouse    0.456315     False
```
TextSim is able to find fuzzy matches of strings ("appl" to "apple" and "icecream" to "ice cream") while not matching "house" to "mouse". Note that TextSim can accept strings of arbitrary length and works on longer texts. You can also perform fuzzy matching within a single list by passing only a single list, e.g. `text_sim.match(queries)`.


### Large-scale Matching and Near-Duplicate Detection Workflow

TextSim offers more complex functionality which allows you to maintain an index of texts (e.g. from a large dataset) and query the index to find similar texts. TextSim supports efficient approximate nearest neighbor (ANN) search using  [USearch](https://github.com/unum-cloud/usearch) which allows it to scale to large datasets with millions of examples.

Please see a minimal working example below for how to use the `.add` and `.search` methods to create and search an index of texts, as well as the demo [colab](notebooks/unisim_text_demo.ipynb) for an in-depth example using TextSim for fuzzy matching on a real-world address matching dataset.

<details>

```python
from unisim import TextSim

text_sim = TextSim(
    store_data=True, # set to False for large datasets to save memory
    index_type="exact", # set to "approx" for large datasets to use ANN search
    batch_size=128, # increasing batch_size on GPU may be faster
    use_accelerator=True, # uses GPU if available, otherwise uses CPU
)

# the dataset can be very large, e.g. millions of texts
dataset = [
    "I love ice cream and cookies",
    "Ice cream is super delicious",
    "my mom makes the best homemade cookies 🍪🍪🍪",
    "This is an example text.",
    "UniSim supports very long texts as well.",
    "UniSim supports multilingual texts too. 你好!",
]

# index your dataset using .add
text_sim.add(index_examples)

# queries can also be a very large dataset
queries = [
    "I luv ice cream and cookies🍦🍪",
    "This is an example query text.",
    "Unrelated text with no match in the dataset..."
]

# search the indexed dataset and find the most similar matches to queries
result_collection = text_sim.search(
    queries,
    similarity_threshold=0.9, # texts match if their similarity >= similarity_threshold
    k=5, # the number of most similar texts in indexed dataset to return for each query
)
```
NOTE: you can set `drop_closest_match=False` in `.search` to ignore the closest match if you know your query exists in the dataset already, e.g. for dataset deduplication, your search queries are the same as your indexed dataset.

NOTE 2: you do not need to add your dataset all at once, you can continously add to and search your index which is useful in production use cases where you have incoming data.

`.search` returns a ResultCollection, which contains the total number of matches found for your queries as well as detailed results containing the most similar matches, their similarity values, and their content. You can visualize the results using `text_sim.visualize(result)`.

```python
# get total matches found across all queries
total_matches = result_collection.total_matches

# visualize a query result (query 0 in this case) in the result_collection
result = result_collection.results[0]
text_sim.visualize(result)
```
`.visualize` prints the following output:
```
Query 0: "I luv ice cream and cookies🍦🍪"
Most similar matches:

  idx  is_match      similarity  text
-----  ----------  ------------  ---------------------------------------------
    0  True                0.91  I love ice cream and cookies
    1  False               0.66  Ice cream is super delicious
    2  False               0.53  my mom makes the best homemade cookies 🍪🍪🍪
    3  False               0.42  This is an example text.
    4  False               0.36  UniSim supports very long texts as well.
```
</details>


## Citing

If you use the UniSim package in your work, please cite:

```bibtex
@software{UniSim_Universal_Similarity_2023,
    title = {{UniSim: Universal Similarity}},
    author = {Marina Zhang, Owen Vallis, and Elie Bursztein},
    url = {https://github.com/google/unisim},
    version = {0.0.1},
    year = {2023}
}

```
Additionally, if you use TextSim or the RETSim model, please cite the following paper: 

```bibtex
@article{RETSim_2023,
    title = {{RETSim: Resilient and Efficient Text Similarity}},
    author = {Marina Zhang, Owen Vallis, Aysegul Bumin, Tanay Vakharia, and Elie Bursztein},
    year = {2023},
    eprint = {TODO}
}
```

## Contributing
To contribute to the project, please check out the [contribution guidelines](CONTRIBUTING.md). Thank you!

## Disclaimer
This is not an official Google product.
