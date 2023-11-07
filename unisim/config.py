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

from .enums import AcceleratorType, BackendType

# internal state
_INTX: str = "int64"
_FLOATX: str = "float32"
_BACKEND: BackendType = BackendType.unknown
_ACCELERATOR: AcceleratorType = AcceleratorType.unknown


def floatx():
    """Returns the default float type, as a string.

    E.g. `'float16'`, `'float32'`, `'float64'`.

    Returns:
        str: the current default float type.
    """
    return _FLOATX


def set_floatx(value):
    """Sets the default float type.

    Note: It is not recommended to set this to float16 for training,
    as this will likely cause numeric stability issues

    Args:
        value (str): `'float16'`, `'float32'`, or `'float64'`.
    """
    global _FLOATX
    if value not in {"float16", "float32", "float64"}:
        raise ValueError("Unknown floatx type: " + str(value))
    _FLOATX = str(value)


def intx():
    """Returns the default int type, as a string.

    E.g. `'int8'`, `'unit8'`, `'int32'`.

    Returns:
        str: the current default int type.
    """
    return _INTX


def set_intx(value):
    """Sets the default int type.

    Args:
        value (str): default int type to use.
        One of `{'int8', 'uint8', 'int16', 'uint16', 'int32', 'int64'}`
    """
    global _INTX
    if value not in {"int8", "uint8", "int16", "uint16", "int32", "int64"}:
        raise ValueError("Unknown intx type: " + str(value))
    _INTX = str(value)


# accelerator
def get_accelerator() -> AcceleratorType:
    "return type of accelerator used"
    return _ACCELERATOR


def set_accelerator(acc: AcceleratorType):
    "Set accelerator type"
    global _ACCELERATOR
    _ACCELERATOR = acc


# backend
def get_backend() -> BackendType:
    """Return the UniSim backend currently in use."""
    return _BACKEND


def set_backend(backend: BackendType):
    """Set UniSim backend to a given framework.

    Args:
        name: Name of the backend, one of {onnx, tensorflow}. Favors Tensorflow
        if available or on GPU, and defaults to Onnx for CPU inference.

    Notes:
        Backend name check is done is `load_backend.py`,
        see `load_backend.py` for the actual loading code.
    """
    global _BACKEND
    _BACKEND = backend
