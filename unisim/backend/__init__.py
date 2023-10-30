"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# - shared function --
# ! Dont import loader related funciton here - will break everything
# ! instead if needed (like set_backend()) put then in root __init__.py
from ..config import floatx  # noqa: F401
from ..config import intx  # noqa: F401
from ..config import set_floatx  # noqa: F401
from ..config import set_intx  # noqa: F401

# - backend specific -
from .load_backend import cosine_similarity, load_model, predict  # noqa: F401
