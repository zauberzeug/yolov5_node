import logging
import multiprocessing
import os

from learning_loop_node import TrainerNode
from learning_loop_node.helpers.entrypoint import node_parser, run_node

from app_code.yolov5_trainer import Yolov5TrainerLogic

parser = node_parser(description='Run the YOLOv5 trainer node')
parser.add_argument('--vram-limit-gb', type=float, default=0,
                    help='Gigabytes of GPU memory the training may use. The batch size is probed against '
                         'this limit instead of the whole card, so a lower limit yields a smaller batch size. '
                         'Use it to share a GPU or to keep headroom against fragmentation. '
                         '0 (default) means no limit.')
args = parser.parse_args()

trainer_logic = Yolov5TrainerLogic(vram_limit_gb=args.vram_limit_gb)
node = TrainerNode(name='Yolov5 Trainer ' + os.uname()[1], trainer_logic=trainer_logic)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    logging.info('using multiprocessing start method %s', multiprocessing.get_start_method())

    run_node('main:node', args)
