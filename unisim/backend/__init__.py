# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# -- shared function --
# ! don't import loader-related functions here - will break
# ! instead, if needed (like set_backend()) put them in root __init__.py
from ..config import floatx  # noqa: F401
from ..config import intx  # noqa: F401
from ..config import set_floatx  # noqa: F401
from ..config import set_intx  # noqa: F401

# -- backend specific --
from .load_backend import cosine_similarity, load_model, predict  # noqa: F401
