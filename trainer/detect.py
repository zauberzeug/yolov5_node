#!/usr/bin/env python3
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch_tensorrt
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size
from utils.torch_utils import select_device

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='model.pt', help='model.pt path')
parser.add_argument('--source', type=str, help='source')  # file/folder, 0 for webcam
args = parser.parse_args()

imgsz = 800
device = select_device('0')  # may be replaced with "cuda"

model = attempt_load(args.weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
model.eval()  # set model to "inference mode"
model.model[-1].export = True  # set Detect() layer export=True

# we need to trace because "script" is not working (see https://nvidia.github.io/Torch-TensorRT/tutorials/creating_torchscript_module_in_python.html)
example_data = torch.randn((1, 3, imgsz, imgsz), device=device)
model = torch.jit.trace(model, example_data)

inputs = [
    torch_tensorrt.Input(
        shape=[1, 3, imgsz, imgsz],
        dtype=torch.half,  # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool)
    ),
]
enabled_precisions = {torch.float, torch.half}  # Run with fp16
ttrt_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)

dataset = LoadImages(args.source, img_size=imgsz, auto_size=64)

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device).half()
    result = ttrt_module(img)  # run inference
    print(result)
