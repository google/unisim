# - shared function --
# ! Dont import loader related funciton here - will break everything
# ! instead if needed (like set_backend()) put then in root __init__.py
from ..config import floatx  # noqa: F401
from ..config import set_floatx  # noqa: F401
from ..config import intx  # noqa: F401
from ..config import set_intx  # noqa: F40

# - backend specific -
# from .load_backend import average_embeddings
# from .load_backend import gather_and_avg
from .load_backend import cosine_similarity
# from .load_backend import knn
from .load_backend import load_model
from .load_backend import predict
