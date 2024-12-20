import numpy as np
import pytest

from unisim.embedder import Embedder
from unisim.unisim import UniSim


class DummyEmbedder(Embedder):
    def __init__(self):
        # Skip parent class init to avoid loading model
        self.batch_size = 2
        self.model_id = "dummy"
        self.verbose = 0

    @property
    def embedding_size(self) -> int:
        return 3

    def embed(self, inputs):
        # Simple mock embedder that returns fixed embeddings
        return np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")[: len(inputs)]

    def predict(self, data):
        # Override predict to avoid using model
        return self.embed(data)


index_type = ["exact", "approx"]


def set_up_test_unisim(index_type):
    unisim = UniSim(
        store_data=True,
        index_type=index_type,
        return_embeddings=True,
        batch_size=2,
        use_accelerator=False,
        model_id="test",
        embedder=DummyEmbedder(),
    )
    # Add some test data - needs to be two items to match the mock embedder
    inputs = ["test1", "test2"]
    unisim.add(inputs)
    return unisim


@pytest.mark.parametrize("index_type", index_type, ids=index_type)
def test_unisim_save_load(index_type, tmp_path):
    # Set up original unisim instance
    unisim = set_up_test_unisim(index_type)

    # Save state to temporary directory
    prefix = str(tmp_path / "unisim_test")
    unisim.save(prefix)

    # Create new instance and restore from saved files
    new_unisim = set_up_test_unisim(index_type)
    new_unisim.load(prefix)

    # Verify search works correctly after restoration
    queries = ["query1"]
    results = new_unisim.search(queries=queries, similarity_threshold=0.5, k=2)

    # Verify results
    assert results.total_matches > 0
    result = results.results[0]
    assert result.query_data == "query1"
    assert len(result.matches) == 2
    assert result.matches[0].data in ["test1", "test2"]
