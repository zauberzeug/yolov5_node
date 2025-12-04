"""
Based on https://github.com/wang-xinyu/tensorrtx/blob/7b79de466c7ac2fcf179e65c2fa4718107f236f9/yolov5/yolov5_det_trt.py
MIT License
"""

import logging
import threading
import time
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pycuda.driver as cuda  # type: ignore # pylint: disable=import-error
import pycuda.gpuarray as gpuarray  # type: ignore # pylint: disable=import-error
import tensorrt as trt  # type: ignore # pylint: disable=import-error
from PIL import Image
from pycuda._driver import (  # type: ignore # pylint: disable=import-error, no-name-in-module
    Error as CudaError,
)
from pycuda.compiler import SourceModule  # type: ignore # pylint: disable=import-error

LEN_ALL_RESULT = 38001
LEN_ONE_RESULT = 38


@dataclass(slots=True, kw_only=True)
class InferenceSlot:
    stream: cuda.Stream
    context: Any  # TODO
    device_preprocess_tmp: gpuarray.GPUArray
    device_input: gpuarray.GPUArray
    device_output: gpuarray.GPUArray
    host_output: np.ndarray


@dataclass(slots=True, kw_only=True)
class BindingDescription:
    name: str
    shape: list[int]
    dtype: np.dtype


Detection = namedtuple('Detection', 'x y w h category probability')


class YoLov5TRT:
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path: str, iou_threshold: float, conf_threshold: float) -> None:
        logging.info('Initializing YOLOv5 TRT engine with iou_threshold: %s, conf_threshold: %s',
                     iou_threshold, conf_threshold)
        # Create a Context on this device,
        try:
            cuda.init()
        except CudaError:
            raise RuntimeError('cuda.init() failed .. detector cannot be used! Try to restart the machine.') from None

        self.cuda_init_error = False

        self.iou_threshold = iou_threshold
        """an iou threshold to filter detections during nms"""
        self.conf_threshold = conf_threshold
        """a confidence threshold to filter detections during nms"""

        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(
                f'Failed to deserialize TensorRT engine from {engine_file_path}. '
                'This is typically caused by a TensorRT version mismatch. '
                'The engine file needs to be rebuilt with the current TensorRT version.'
            )

        self._init_engine_IO_binding_descriptions(engine)

        batch_size = self.input_binding.shape[0]
        assert batch_size == 1

        self.inference_slots: list[InferenceSlot] = []

        # Store
        self.engine = engine

    @contextmanager
    def _active_cuda_context(self):
        assert self.ctx is not None
        self.ctx.push()
        try:
            yield
        finally:
            self.ctx.pop()

    def _init_engine_IO_binding_descriptions(self, engine: trt.ICudaEngine) -> None:
        """
        Set up TensorRT bindings for input and output tensors.
        For YoloV5, there should be exactly one input and one output.

        :param engine: TensorRT CUDA engine
        """
        input_descriptions = []
        output_descriptions = []

        for binding_name in engine:
            shape = list(engine.get_tensor_shape(binding_name))
            print('binding_name:', binding_name, shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding_name))

            binding_description = BindingDescription(name=binding_name, shape=shape, dtype=dtype)
            if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                input_descriptions.append(binding_description)
            elif engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                output_descriptions.append(binding_description)
            else:
                print(f'unknown binding: "{binding_name}"')

        assert len(input_descriptions) == 1
        assert len(output_descriptions) == 1

        self.input_binding = input_descriptions[0]
        self.output_binding = output_descriptions[0]

    @property
    def input_w(self) -> int:
        return self.input_binding.shape[-1]

    @property
    def input_h(self) -> int:
        return self.input_binding.shape[-2]

    def _create_inference_slot(self) -> InferenceSlot:
        input_img_shape = [self.input_h, self.input_w, 3]
        device_preprocess_tmp = gpuarray.empty(shape=input_img_shape, dtype=np.uint8)
        device_input = gpuarray.empty(shape=self.input_binding.shape, dtype=self.input_binding.dtype)
        device_output = gpuarray.empty(shape=self.output_binding.shape, dtype=self.output_binding.dtype)
        host_output = cuda.pagelocked_empty(self.output_binding.shape, self.output_binding.dtype)

        return InferenceSlot(
            stream=cuda.Stream(),
            context=self.engine.create_execution_context(),
            device_preprocess_tmp=device_preprocess_tmp,
            device_input=device_input,
            device_output=device_output,
            host_output=host_output,
        )

    def _preprocess_image(self, image_raw: np.ndarray, slot: InferenceSlot) -> None:
        """Resize and pad it to target size, normalize to [0,1], transform to NCHW format.

        param image_raw: A raw image ndarray, shape is [H,W,3]
        param slot: An InferenceSlot object whose device_input will be updated
        """
        h, w, _ = image_raw.shape
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        if tw == w and th == h:
            image = image_raw
        else:
            pil_image = Image.fromarray(image_raw)
            pil_image = pil_image.resize((tw, th), Image.Resampling.BILINEAR)
            image = np.array(pil_image)
        # Pad with (128, 128, 128)
        if tx1 != 0 or tx2 != 0 or ty1 != 0 or ty2 != 0:
            image = np.pad(
                image,
                ((ty1, ty2), (tx1, tx2), (0, 0)),
                mode='constant',
                constant_values=128
            )
        slot.device_preprocess_tmp.set_async(image, stream=slot.stream)
        assert image.shape == slot.device_preprocess_tmp.shape, f'{image.shape}'
        assert slot.device_preprocess_tmp.dtype == np.uint8
        assert slot.device_input.dtype == np.float32
        _convert_hwc_uint8_to_nchw_float(slot.device_preprocess_tmp, slot.device_input, stream=slot.stream)

    def _dispatch_gpu_inference(self, inference_slot: InferenceSlot):
        # Run inference.
        context = inference_slot.context
        context.set_tensor_address(self.input_binding.name, inference_slot.device_input.ptr)
        context.set_tensor_address(self.output_binding.name, inference_slot.device_output.ptr)
        context.execute_async_v3(stream_handle=inference_slot.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(inference_slot.host_output, inference_slot.device_output.ptr, inference_slot.stream)

    # ============================================ PUBLIC METHODS ============================================

    def infer(self, image_raw: np.ndarray) -> tuple[list[Detection], float]:
        threading.Thread.__init__(self)  # type: ignore

        with self._active_cuda_context():
            inference_slot = self.inference_slots.pop() if self.inference_slots else self._create_inference_slot()

            # Do image preprocess
            origin_h, origin_w, _ = image_raw.shape
            self._preprocess_image(image_raw, inference_slot)

            # Run inference
            start = time.time()
            self._dispatch_gpu_inference(inference_slot)
            # Synchronize the stream
            inference_slot.stream.synchronize()
            end = time.time()

        # Do postprocess
        post_process_results = self._post_process(inference_slot.host_output, origin_h, origin_w)
        detections = _pack_detection_results(*post_process_results)

        self.inference_slots.append(inference_slot)
        return detections, end - start

    def infer_batch(self, images: list[np.ndarray]) -> tuple[list[list[Detection]], float]:
        threading.Thread.__init__(self)  # type: ignore
        assert self.ctx is not None

        with self._active_cuda_context():
            # Claim slots (and possibly create) slots for all images in the batch
            inference_slots = [self.inference_slots.pop() if self.inference_slots else self._create_inference_slot()
                               for _ in images]

            start = time.time()
            for image, inference_slot in zip(images, inference_slots, strict=True):
                self._preprocess_image(image, inference_slot)
                self._dispatch_gpu_inference(inference_slot)

            end = time.time()

            results = []
            for image, inference_slot in zip(images, inference_slots, strict=True):
                h, w, _ = image.shape
                # Synchronize stream to make sure that inference_slot.host_output has valid values
                inference_slot.stream.synchronize()
                post_process_results = self._post_process(inference_slot.host_output, h, w)
                detections = _pack_detection_results(*post_process_results)
                results.append(detections)
                self.inference_slots.append(inference_slot)

        return results, end - start

    def destroy(self) -> None:
        """
        Destroy the TRT engine and free up resources.

        Advanced destroy function to clean up TRT objects.
        This differs from the original implementation in TensorRTX where only the context was popped.
        We had to apply this modification to avoid a segmentation fault on re-initialization
        (particularly when setting yolov5=None).
        """

        # If init failed or already cleaned up, nothing to do
        if getattr(self, "ctx", None) is None:
            return

        with self._active_cuda_context():
            # 1) Free device buffers
            for slot in getattr(self, "inference_slots", []):
                slot.stream.synchronize()
                try:
                    slot.device_preprocess_tmp.free()
                except Exception:
                    pass
                try:
                    slot.device_input.free()
                except Exception:
                    pass
                try:
                    slot.device_output.free()
                except Exception:
                    pass
                try:
                    del slot.context
                except Exception:
                    pass

            # 2) Destroy TRT objects while context is current
            try:
                del self.engine
            except Exception:
                pass

            # 3) Drop slots and associated ressources
            self.inference_slots = []

        # 4) Drop context
        try:
            self.ctx.detach()  # type: ignore
        except Exception:
            pass
        self.ctx = None

    def xywh2xyxy(self, origin_h: int, origin_w: int, x: np.ndarray) -> np.ndarray:
        """
        Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def _post_process(self, output_tensor: np.ndarray, origin_h: int, origin_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """

        output = output_tensor[0, 0:LEN_ALL_RESULT, 0, 0]
        num = int(output[0])  # Get the num of boxes detected
        # Reshape to a 2D ndarray
        pred = np.reshape(output[1:], (-1, LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]
        # Do nms
        boxes = self._non_max_suppression(pred, origin_h, origin_w)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1: np.ndarray, box2: np.ndarray, x1y1x2y2: bool = True) -> np.ndarray:
        """
        Compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
            np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def _non_max_suppression(self, prediction: np.ndarray, origin_h: int, origin_w: int) -> np.ndarray:
        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.

        param prediction:  detections, (x1, y1, x2, y2, conf, cls_id)
        param origin_h: original image height
        param origin_w: original image width

        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """

        conf_thres = self.conf_threshold
        nms_thres = self.iou_threshold

        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]

        if len(confs) == 0:
            return boxes

        max_class = int(np.max(boxes[:, -1]))
        num_classes = max_class + 1

        # Perform per-class non-maximum suppression
        keep_boxes = []
        for _class in range(num_classes):
            class_boxes = boxes[boxes[:, -1] == _class]
            while class_boxes.shape[0]:
                # Indices of boxes with lower confidence scores, large IOUs
                large_overlap = self.bbox_iou(np.expand_dims(class_boxes[0, :4], 0), class_boxes[:, :4]) > nms_thres
                keep_boxes += [class_boxes[0]]
                class_boxes = class_boxes[~large_overlap]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


def _pack_detection_results(
        result_boxes: np.ndarray, result_scores: np.ndarray, result_classid: np.ndarray) -> list[Detection]:
    """Pack detection array results into a list of Detection namedtuples, suitable for returning from node"""
    detections = []
    for j, box in enumerate(result_boxes):
        x, y, br_x, br_y = box
        w = br_x - x
        h = br_y - y
        detections.append(Detection(int(x), int(y), int(w), int(h),
                                    int(result_classid[j]), round(float(result_scores[j]), 2)))
    return detections


hwc_uint8_to_nchw_float_kernel: None | SourceModule = None


def _convert_hwc_uint8_to_nchw_float(src: gpuarray.GPUArray, dst: gpuarray.GPUArray, stream: cuda.Stream) -> None:
    """Convert image from HWC uint8 [0,255] to NCHW float32 [0.0,1.0]"""
    global hwc_uint8_to_nchw_float_kernel
    if hwc_uint8_to_nchw_float_kernel is None:
        hwc_uint8_to_nchw_float_kernel = SourceModule("""
        __global__ void hwc_to_nchw(unsigned char *src, float *dst,
                                    int H, int W, int C)
        {
            int h = blockIdx.y * blockDim.y + threadIdx.y;
            int w = blockIdx.x * blockDim.x + threadIdx.x;
            if (h < H && w < W)
            {
                for(int c = 0; c < C; ++c)
                {
                    int src_idx = h*W*C + w*C + c;     // NHWC
                    int dst_idx = c*H*W + h*W + w;     // NCHW
                    dst[dst_idx] = float(src[src_idx]) / 255.0; // uint8 -> float conversion
                }
            }
        }
        """)

    H, W, C = src.shape
    func = hwc_uint8_to_nchw_float_kernel.get_function("hwc_to_nchw")  # type: ignore

    block = (16, 16, 1)
    grid = ((W + 15)//16, (H + 15)//16)

    func(
        src.gpudata,
        dst.gpudata,
        np.int32(H), np.int32(W), np.int32(C),
        block=block, grid=grid, stream=stream
    )


class WarmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper: YoLov5TRT) -> None:
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self) -> None:
        zeros = np.zeros([self.yolov5_wrapper.input_h, self.yolov5_wrapper.input_w, 3], dtype=np.uint8)
        _, use_time = self.yolov5_wrapper.infer(zeros)
        print(f'warm_up time->{use_time * 1000:.2f}ms')
