"""
Original from https://github.com/ultralytics/yolov5/blob/master/export.py
 GPL-3.0 License
"""


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Usage:
    $ python path/to/export.py --weights yolov5s.pt --include engine ...


"""


# from utils.activations import SiLU
# from models.yolo import Detect
# from models.experimental import attempt_load
# from models.common import Conv
# import argparse
# import os
# import sys
# import time
# from pathlib import Path
# import torch
# import torch.nn as nn
# import logging
# from yolov5TRT.torch_utils import select_device
# from yolov5TRT.general import colorstr, file_size, check_img_size
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
#     # YOLOv5 ONNX export
#     try:
#         import onnx

#         logging.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
#         f = file.with_suffix('.onnx')

#         torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
#                           training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
#                           do_constant_folding=not train,
#                           input_names=['images'],
#                           output_names=['output'],
#                           dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
#                                         'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
#                                         } if dynamic else None)

#         # Checks
#         model_onnx = onnx.load(f)  # load onnx model
#         onnx.checker.check_model(model_onnx)  # check onnx model
#         # LOGGER.info(onnx.helper.printable_graph(model_onnx.graph))  # print

#         # Simplify
#         if simplify:
#             try:
#                 import onnxsim

#                 logging.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
#                 model_onnx, check = onnxsim.simplify(
#                     model_onnx,
#                     dynamic_input_shape=dynamic,
#                     input_shapes={'images': list(im.shape)} if dynamic else None)
#                 assert check, 'assert check failed'
#                 onnx.save(model_onnx, f)
#             except Exception as e:
#                 logging.info(f'{prefix} simplifier failure: {e}')
#         logging.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
#         return f
#     except Exception as e:
#         logging.info(f'{prefix} export failure: {e}')


# def export_engine(model, im, file, train, half, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
#     # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
#     try:
#         import tensorrt as trt

#         if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
#             grid = model.model[-1].anchor_grid
#             model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
#             export_onnx(model, im, file, 12, train, False, simplify)  # opset 12
#             model.model[-1].anchor_grid = grid
#         else:  # TensorRT >= 8
#             export_onnx(model, im, file, 13, train, False, simplify)  # opset 13
#         onnx = file.with_suffix('.onnx')

#         logging.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
#         assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
#         assert onnx.exists(), f'failed to export ONNX file: {onnx}'
#         f = file.with_suffix('.engine')  # TensorRT engine file
#         logger = trt.Logger(trt.Logger.INFO)
#         if verbose:
#             logger.min_severity = trt.Logger.Severity.VERBOSE

#         builder = trt.Builder(logger)
#         config = builder.create_builder_config()
#         config.max_workspace_size = workspace * 1 << 30

#         flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#         network = builder.create_network(flag)
#         parser = trt.OnnxParser(network, logger)
#         if not parser.parse_from_file(str(onnx)):
#             raise RuntimeError(f'failed to load ONNX file: {onnx}')

#         inputs = [network.get_input(i) for i in range(network.num_inputs)]
#         outputs = [network.get_output(i) for i in range(network.num_outputs)]
#         logging.info(f'{prefix} Network Description:')
#         for inp in inputs:
#             logging.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
#         for out in outputs:
#             logging.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

#         half &= builder.platform_has_fast_fp16
#         logging.info(f'{prefix} building FP{16 if half else 32} engine in {f}')
#         if half:
#             config.set_flag(trt.BuilderFlag.FP16)
#         with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
#             t.write(engine.serialize())
#         logging.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
#         return f
#     except Exception as e:
#         logging.info(f'\n{prefix} export failure: {e}')


# @torch.no_grad()
# def run(
#         weights=ROOT / 'yolov5s.pt',  # weights path
#         imgsz=(832, 832),  # image (height, width)
#         batch_size=1,  # batch size
#         device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         include=('onnx'),  # include formats
#         half=False,  # FP16 half-precision export
#         inplace=False,  # set YOLOv5 Detect() inplace=True
#         train=False,  # model.train() mode
#         dynamic=False,  # ONNX/TF: dynamic axes
#         simplify=False,  # ONNX: simplify model
#         opset=12,  # ONNX: opset version
#         verbose=False,  # TensorRT: verbose log
#         workspace=4,  # TensorRT: workspace size (GB)
# ):
#     t = time.time()
#     include = [x.lower() for x in include]  # to lowercase
#     file = Path(weights)  # PyTorch weights

#     # Load PyTorch model
#     device = select_device(device)
#     assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
#     model = attempt_load(weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
#     nc, names = model.nc, model.names  # number of classes, class names

#     # Checks
#     imgsz *= 2 if len(imgsz) == 1 else 1  # expand
#     opset = 12 if ('openvino' in include) else opset  # OpenVINO requires opset <= 12
#     assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

#     # Input
#     gs = int(max(model.stride))  # grid size (max stride)
#     imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
#     im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

#     # Update model
#     if half:
#         im, model = im.half(), model.half()  # to FP16
#     model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
#     for k, m in model.named_modules():
#         if isinstance(m, Conv):  # assign export-friendly activations
#             if isinstance(m.act, nn.SiLU):
#                 m.act = SiLU()
#         elif isinstance(m, Detect):
#             m.inplace = inplace
#             m.onnx_dynamic = dynamic
#             if hasattr(m, 'forward_export'):
#                 m.forward = m.forward_export  # assign custom forward (optional)

#     for _ in range(2):
#         y = model(im)  # dry runs
#     shape = tuple(y[0].shape)  # model output shape
#     logging.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

#     # TensorRT required before ONNX
#     export_engine(model, im, file, train, half, simplify, workspace, verbose)

#     # Finish
#     f = [str(x) for x in f if x]  # filter out '' and None
#     if any(f):
#         logging.info(f'\nExport complete ({time.time() - t:.2f}s)'
#                      f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
#                      f"\nDetect:          python detect.py --weights {f[-1]}"
#                      f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
#                      f"\nValidate:        python val.py --weights {f[-1]}"
#                      f"\nVisualize:       https://netron.app")
#     return f  # return list of exported files/dirs


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
#     parser.add_argument('--batch-size', type=int, default=1, help='batch size')
#     parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
#     parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
#     parser.add_argument('--train', action='store_true', help='model.train() mode')
#     parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
#     parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
#     parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
#     parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
#     parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
#     parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
#     parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
#     parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
#     parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
#     parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
#     parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
#     parser.add_argument('--include', nargs='+',
#                         default=['onnx'],
#                         help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
#     opt = parser.parse_args()
#     return opt


# def main(opt):
#     for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
#         run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
