import logging
import multiprocessing
import os

from learning_loop_node import TrainerNode
from learning_loop_node.helpers.entrypoint import node_parser, run_node

from app_code.yolov5_trainer import Yolov5TrainerLogic

parser = node_parser(description='Run the YOLOv5 trainer node', vram_limit=True)
parser.add_argument('--image-cache', choices=['ram', 'off'], default='ram',
                    help='Keep the decoded images in RAM for the whole training, as far as half of the free memory '
                         'allows; "off" decodes every image anew in each epoch')
args = parser.parse_args()

trainer_logic = Yolov5TrainerLogic(vram_limit_gb=args.vram_limit_gb, cache_images=args.image_cache == 'ram')
node = TrainerNode(name='Yolov5 Trainer ' + os.uname()[1], trainer_logic=trainer_logic)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    logging.info('using multiprocessing start method %s', multiprocessing.get_start_method())

    run_node('main:node', args)
