import os

from learning_loop_node import DetectorNode
from learning_loop_node.helpers.entrypoint import node_parser, run_node

from yolov5_detector import Yolov5DetectorParams

MAX_BUILDER_OPTIMIZATION_LEVEL = 5

parser = node_parser(description='Run the YOLOv5 detector node')
parser.add_argument('--weight-type', default='FP16', choices=['FP16', 'FP32', 'INT8'],
                    help='Inference weight type')
parser.add_argument('--builder-optimization-level', type=int, default=4,
                    choices=range(MAX_BUILDER_OPTIMIZATION_LEVEL + 1),
                    help='TensorRT builder optimization level; higher levels search more kernel '
                         'tactics for potentially faster inference at the cost of longer engine builds')
parser.add_argument('--iou-threshold', type=float, default=0.45, help='Threshold for non-maximum-suppression')
parser.add_argument('--conf-threshold', type=float, default=0.2, help='Minimum confidence for detections')

args = parser.parse_args()

params = Yolov5DetectorParams(
    weight_type=args.weight_type,
    iou_threshold=args.iou_threshold,
    conf_threshold=args.conf_threshold,
    builder_optimization_level=args.builder_optimization_level,
)
node = DetectorNode(name='YOLOv5 Detector ' + os.uname()[1], detector_factory=params)

if __name__ == '__main__':
    run_node('main:node', args)
