import os

from learning_loop_node import DetectorNode
from learning_loop_node.helpers.entrypoint import node_parser, run_node

from yolov5_detector import Yolov5DetectorParams

parser = node_parser(description='Run the YOLOv5 CPU detector node')
parser.add_argument('--iou-threshold', type=float, default=0.45, help='Threshold for non-maximum-suppression')
parser.add_argument('--conf-threshold', type=float, default=0.2, help='Minimum confidence for detections')

args = parser.parse_args()

params = Yolov5DetectorParams(iou_threshold=args.iou_threshold, conf_threshold=args.conf_threshold)
node = DetectorNode(name='YOLOv5 CPU Detector ' + os.uname()[1], detector_factory=params)

if __name__ == '__main__':
    run_node('main:node', args)
