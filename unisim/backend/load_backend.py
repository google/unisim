import os
from ..config import set_backend, get_backend
from ..config import set_accelerator, get_accelerator, AcceleratorType

# override existing setting to allow reload
if 'BACKEND' in os.environ:
    set_backend(os.environ['BACKEND'])
elif not get_backend():
    # check if we find jax
    try:
        import tensorflow as tf  # noqa: F403, F401
    except ImportError:
        set_backend('onnx')
    else:
        set_backend('tensorflow')

if get_backend() == 'onnx':
    from .onnx import *  # noqa: F403, F401
    # FIXME onnx accelerator type support
    set_accelerator(AcceleratorType.cpu)

elif get_backend() == 'tensorflow':
    from .tf import *  # noqa: F403, F401
    # ensure we don't run out of memory
    # FIXME: adjust for GPU vs CPU and print it
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
    raise ValueError('Unknown Backend')

print(f'Using {get_backend()} backend on device: {get_accelerator().name}')
