
import argparse
import os

from app_code.yolov5 import generate_wts


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'

    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')

    elif os.path.isdir(args.output):
        args.output = os.path.join(args.output, os.path.splitext(os.path.basename(args.weights))[0] + '.wts')

    return args.weights, args.output


if __name__ == '__main__':
    pt_file_path, wts_file_path = parse_args()
    generate_wts(pt_file_path, wts_file_path)
