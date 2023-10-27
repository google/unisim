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

import os
from ..config import set_backend, get_backend
from ..config import set_accelerator, get_accelerator
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
    # FIXME detect multi
    # detecting which devices we are using
    devices_types = [d.device_type for d in tf.config.list_physical_devices()]

    if "GPU" in devices_types:
        set_accelerator(AcceleratorType.gpu)
    else:
        set_accelerator(AcceleratorType.cpu)

else:
    set_accelerator(AcceleratorType.cpu)

# choose backend
accel = get_accelerator()
backend = get_backend()
if backend != BackendType.unknown:
    # backend forced by the user
    pass
elif accel == AcceleratorType.cpu:
    # on CPU always onnx
    set_backend(BackendType.onnx)
elif TF_AVAILABLE:
    # potentially revisit
    set_backend(BackendType.tf)
else:
    # default to onnx
    set_backend(BackendType.onnx)

# post detection
if get_backend() == BackendType.onnx:
    from .onnx import *  # noqa: F403, F401

    # FIXME onnx accelerator type support
    set_accelerator(AcceleratorType.cpu)

elif get_backend() == BackendType.tf:
    from .tf import *  # noqa: F403, F401

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

print("Loaded backend")
print(f"Using {get_backend().name} with {get_accelerator().name}")
