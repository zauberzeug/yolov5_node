#! /usr/bin/env python3
import sys

from livesync import Folder, sync

assert len(sys.argv) == 2, 'Usage: python livesync.py <robot>'
arg = sys.argv[1]
base_dir = '~'
robot = arg


sync(
    Folder('.', f'{robot}:{base_dir}/yolov5_livesynced/yolov5_node'),
    Folder('../learning_loop_node', f'{robot}:{base_dir}/yolov5_livesynced/learning_loop_node'),
)
