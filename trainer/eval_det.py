"""The final evaluation of a training's best model: the per-class table, the confusion matrix and the curve data.

The node runs it after `train_det.py` has ended, whether the training finished or was stopped, so a stopped
training gets its evaluation too. Everything it needs comes from the result folder `train_det.py` wrote:
`opt.yaml`, `hyp.yaml`, the measured batch size and `weights/best.pt`.

Usage:
    $ python eval_det.py --result-dir <training folder>/result
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from learning_loop_node.trainer.cuda import add_vram_limit_argument, limit_cuda_memory

from app_code import training_recipe, training_run
from app_code.yolov5 import val as validate
from app_code.yolov5.models.experimental import attempt_load
from app_code.yolov5.utils.dataloaders import create_dataloader
from app_code.yolov5.utils.general import LOGGER, check_dataset, check_img_size, colorstr
from app_code.yolov5.utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]/'app_code/yolov5'  # yolov5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def evaluate(result_dir: Path) -> None:
    best = result_dir / 'weights' / 'best.pt'
    if not best.exists():
        LOGGER.info(f'No {best} to evaluate')
        return
    opt = argparse.Namespace(**yaml.safe_load((result_dir / 'opt.yaml').read_text()))
    hyp = yaml.safe_load((result_dir / 'hyp.yaml').read_text())
    batch_size = training_run.TrainingRun(result_dir).batch_size or opt.batch_size
    point_sizes_by_id = {int(k): float(v) for k, v in (item.split(':')
                         for item in str(opt.point_sizes_by_id).split(','))} if opt.point_sizes_by_id else {}

    data_dict = check_dataset(opt.data)
    # the final evaluation reports on the test set at the confidence picked for the best model;
    # the test set may be the validation folder
    val_path, test_path = data_dict['val'], data_dict['test']
    model_metadata = torch.load(best, map_location='cpu', weights_only=False).get('model_metadata', {})
    report_conf = model_metadata.get('conf_threshold')

    device = select_device(opt.device, batch_size=batch_size)
    model = attempt_load(best, device).half()
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    dataloader = create_dataloader(test_path,
                                   imgsz,
                                   batch_size // 2,
                                   gs,
                                   opt.single_cls,
                                   hyp=hyp,
                                   rect=True,
                                   rank=-1,
                                   workers=opt.workers * 2,
                                   pad=training_recipe.VAL_PAD,
                                   prefix=colorstr('test: '),
                                   point_sizes_by_id=point_sizes_by_id)[0]

    LOGGER.info(f"\n{'Validating' if test_path == val_path else 'Testing'} {best}...")
    validate.run(data_dict,
                 batch_size=batch_size // 2,
                 imgsz=imgsz,
                 conf_thres=opt.conf_thres,
                 iou_thres=opt.iou_thres,
                 model=model,
                 single_cls=opt.single_cls,
                 dataloader=dataloader,
                 task='test',
                 save_dir=result_dir,
                 verbose=True,
                 plots=False,
                 report_conf=report_conf)


if __name__ == '__main__':
    print('START: YOLOv5 - final evaluation', flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True, help='the result folder of train_det.py')
    add_vram_limit_argument(parser)
    args = parser.parse_args()
    limit_cuda_memory(args.vram_limit_gb)
    evaluate(args.result_dir)
    torch.cuda.empty_cache()
    print('END: YOLOv5 - final evaluation', flush=True)
