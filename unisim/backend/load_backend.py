# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os

from ..config import get_accelerator, get_backend, set_accelerator, set_backend
from ..enums import AcceleratorType, BackendType

TF_AVAILABLE = False

# override existing setting to allow reload
if "BACKEND" in os.environ:
    bs = os.environ["BACKEND"]
    if bs == "onnx":
        set_backend(BackendType.onnx)
    elif bs == "tf":
        set_backend(BackendType.tf)
        import tensorflow as tf
    else:
        raise ValueError(f"Unknown environment backend {bs}")
elif not get_backend() or get_backend() == BackendType.unknown:
    # check if we find tensorflow
    try:
        import tensorflow as tf  # noqa: F403, F401

        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

# detect accelerator
if TF_AVAILABLE or get_backend() == BackendType.tf:
    devices_types = [d.device_type for d in tf.config.list_physical_devices()]

    if "GPU" in devices_types:
        set_accelerator(AcceleratorType.gpu)
    else:
        set_accelerator(AcceleratorType.cpu)

else:
    set_accelerator(AcceleratorType.cpu)

# choose backend if not set by user
accel = get_accelerator()
backend = get_backend()

if "BACKEND" not in os.environ:
    if backend != BackendType.unknown:
        # backend forced by the user
        pass
    elif accel == AcceleratorType.cpu:
        # on CPU always onnx
        set_backend(BackendType.onnx)
    elif TF_AVAILABLE and accel == AcceleratorType.gpu:
        # on GPU use TF by default
        set_backend(BackendType.tf)
    else:
        # if TensorFlow not available
        set_backend(BackendType.onnx)

# post detection
if get_backend() == BackendType.onnx:
    from .onnx import *  # noqa: F403, F401

    # FIXME(marinazh): onnx accelerator type support
    set_accelerator(AcceleratorType.cpu)

elif get_backend() == BackendType.tf:
    from .tf import *  # type: ignore # noqa: F403, F401

    # ensure we are not running out of memory
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

else:
    raise ValueError(f"Unknown backend {get_backend()}")

logging.info("Loaded backend")
logging.info("Using %s with %s", get_backend().name.upper(), get_accelerator().name.upper())
