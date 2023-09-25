import os
from ..config import set_backend, get_backend
from ..config import set_accelerator, get_accelerator
from ..enums import AcceleratorType, BackendType
TF_AVAILABLE = False
# override existing setting to allow reload
if 'BACKEND' in os.environ:
    bs = os.environ['BACKEND']
    if bs == 'onnx':
        set_backend(BackendType.onnx)
    elif bs == 'tf':
        set_backend(BackendType.tf)
    else:
        raise ValueError(f"Unknown environement backend {bs}")
elif not get_backend() or get_backend() == BackendType.unknown:
    # check if we find jax
    try:
        import tensorflow as tf  # noqa: F403, F401
    except ImportError:
        TF_AVAILABLE = False
    else:
        TF_AVAILABLE = True

# detect device
if TF_AVAILABLE:
    # FIXME detect multi
    # detecting which devices we are using
    device = tf.config.list_physical_devices()[0].device_type
    if device == "CPU":
        set_accelerator(AcceleratorType.cpu)
    elif device == "GPU":
        set_accelerator(AcceleratorType.gpu)
    elif device == "TPU":
        set_accelerator(AcceleratorType.tpu)
    else:
        set_accelerator(AcceleratorType.unknown)
else:
    # FIXME onnx gpu detection
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
elif tf:
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
    raise ValueError(f'Unknown Backend {get_backend()}')

print(f'Using {get_backend().name} with {get_accelerator().name}')
