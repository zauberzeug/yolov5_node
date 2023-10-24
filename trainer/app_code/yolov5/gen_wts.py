import argparse
import os
import struct

import torch

if __name__ == '__main__':
    from utils.torch_utils import select_device
else:
    from .utils.torch_utils import select_device

# wget https://raw.githubusercontent.com/wang-xinyu/tensorrtx/master/yolov5/gen_wts.py


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-o', '--output', help='Output (.wts) file path (optional)')
    parser.add_argument(
        '-t', '--type', type=str, default='detect', choices=['detect', 'cls', 'seg'],
        help='determines the model is detection/classification')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args.weights, args.output, args.type


def gen_wts(pt_file_path: str, wts_file_path: str, model_type: str='detect'):
    print(f'Generating .wts for {model_type} model')

    # Load model
    print(f'Loading {pt_file_path}')
    device = select_device('cpu')
    model = torch.load(pt_file_path, map_location=device)  # Load FP32 weights
    model = model['ema' if model.get('ema') else 'model'].float()

    if model_type in ['detect', 'seg']:
        # update anchor_grid info
        anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
        # model.model[-1].anchor_grid = anchor_grid
        delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
        # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
        model.model[-1].register_buffer("anchor_grid", anchor_grid)
        model.model[-1].register_buffer("strides", model.model[-1].stride)

    model.to(device).eval()

    print(f'Writing into {wts_file_path}')
    with open(wts_file_path, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

if __name__ == '__main__':
    pt_file, wts_file, m_type = parse_args()
    gen_wts(pt_file, wts_file, m_type)
