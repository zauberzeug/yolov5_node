import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
import logging
from yolov5TRT.general import check_version
from collections import namedtuple, OrderedDict
import torch
import numpy as np
import torch.nn as nn


class TensorRTBackend(nn.Module):
    def __init__(self, weights, device, data):
        logging.info(f'Loading {weights} for TensorRT inference...')
        check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        batch_size = bindings['images'].shape[0]
        self.__dict__.update(locals())  # assign all variables to self
