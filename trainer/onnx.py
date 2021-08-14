import torch
import logging
from yolor.models.models import Darknet


def export(pytorch_model: str):
    resolution = 1280
    model = Darknet(pytorch_model.replace('.pt', '.cfg'), resolution).cuda()
    device = torch.device('cuda:0')
    model.load_state_dict(torch.load(pytorch_model, map_location=device)['model'])
    model.half()  # FP16

    # set the model to inference mode
    model.to(device).eval()

    dummy_input = torch.randn(1, 3, resolution, resolution, device=device).half()
    target_filename = pytorch_model.replace('.pt', '.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        target_filename,
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['modelInput'],             # the model's input names
        output_names=['modelOutput'],           # the model's output names
        dynamic_axes={
            'modelInput': {0: 'batch_size'},    # variable length axes
            'modelOutput': {0: 'batch_size'}})

    logging.info('Model has been converted to ONNX')
    return target_filename
