"""The final evaluation of a training's best model on the test set: per-class table, confusion matrix, curve data.

The only writer of `confusion_matrix_test.json` and `curves_test.json`; the test pass in `train_det.py` only counts.

Reads `opt.yaml`, `hyp.yaml`, the measured batch size and `weights/best.pt` from the result folder of `train_det.py`.

Usage:
    $ python eval_det.py --result-dir <training folder>/result
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from learning_loop_node.trainer.cuda import add_vram_limit_argument, limit_cuda_memory

from app_code import training_recipe, training_run
from app_code.yolov5 import val as validate
from app_code.yolov5.models.experimental import attempt_load
from app_code.yolov5.utils.general import LOGGER, check_dataset, check_img_size
from app_code.yolov5.utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]/'app_code/yolov5'  # yolov5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def evaluate(result_dir: Path) -> None:
    best = result_dir / 'weights' / 'best.pt'
    if not best.exists():
        LOGGER.info('No %s to evaluate', best)
        return
    opt = argparse.Namespace(**yaml.safe_load((result_dir / 'opt.yaml').read_text()))
    hyp = yaml.safe_load((result_dir / 'hyp.yaml').read_text())
    batch_size = training_run.TrainingRun(result_dir).batch_size or opt.batch_size

    data_dict = check_dataset(opt.data)
    # the test set may be the validation folder
    val_path, test_path = data_dict['val'], data_dict['test']
    model_metadata = torch.load(best, map_location='cpu', weights_only=False).get('model_metadata', {})
    report_conf = model_metadata.get('conf_threshold')
    if report_conf is None:
        # such a model found nothing on the validation images and was never reported; without the confidence,
        # the confusion matrix and the curves would be read at two different ones
        LOGGER.info('No confidence was picked for %s, nothing to evaluate', best)
        return

    device = select_device(opt.device, batch_size=batch_size)
    model = attempt_load(best, device).half()
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    dataloader = training_recipe.evaluation_loader(
        test_path, 'test', img_size=imgsz, batch_size=batch_size, stride=gs, single_cls=opt.single_cls, hyp=hyp,
        workers=opt.workers, point_sizes_by_id=training_recipe.parse_point_sizes(opt.point_sizes_by_id))

    LOGGER.info('\n%s %s...', 'Validating' if test_path == val_path else 'Testing', best)
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
                 report_conf=report_conf,
                 keep=True)


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
