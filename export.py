import os
import torch
import argparse

from easy_ViTPose.vit_models.model import ViTPose
from easy_ViTPose.vit_utils.util import infer_dataset_by_path, dyn_model_import
import onnx
from onnxsim import simplify
from sbi4onnx import initialize

parser = argparse.ArgumentParser()
# parser.add_argument('--model-ckpt', type=str, required=True, help='The torch model that shall be used for conversion')
# parser.add_argument('--model-name', type=str, required=True, choices=['s', 'b', 'l', 'h'], help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
parser.add_argument('--output', type=str, default='ckpts/', help='File (without extension) or dir path for checkpoint output')
parser.add_argument('--dataset', type=str, required=False, default=None, help='Name of the dataset. If None it"s extracted from the file name. ["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
args = parser.parse_args()

MODEL_TYPES = [
    's',
    'b',
    'l',
    'h',
]

CKPTS = [
    "vitpose-{MODEL_TYPE}-aic.pth",
    "vitpose-{MODEL_TYPE}-ap10k.pth",
    "vitpose-{MODEL_TYPE}-apt36k.pth",
    "vitpose-{MODEL_TYPE}-coco_25.pth",
    "vitpose-{MODEL_TYPE}-coco.pth",
    "vitpose-{MODEL_TYPE}-mpii.pth",
    "vitpose-{MODEL_TYPE}-wholebody.pth",
]

for model_type in MODEL_TYPES:
    for ckpt_file in CKPTS:
        ckpt_file = f'{ckpt_file.replace("{MODEL_TYPE}", model_type)}'

        # Get dataset and model_cfg
        dataset = args.dataset
        if dataset is None:
            dataset = infer_dataset_by_path(ckpt_file)
        assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], 'The specified dataset is not valid'
        model_cfg = dyn_model_import(dataset, model_type)

        # Convert to onnx and save
        print('>>> Converting to ONNX')
        CKPT_PATH = ckpt_file
        C, H, W = (3, 256, 192)

        model = ViTPose(model_cfg)

        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        model.load_state_dict(ckpt)
        model.eval()

        input_names = ["input"]
        output_names = ["output"]

        device = next(model.parameters()).device
        inputs = torch.randn(1, C, H, W).to(device)

        # dynamic_axes = {
        #     'input': {0: 'batch'},
        #     'output': {0: 'batch'}
        # }

        out_name = os.path.basename(ckpt_file).replace('.pth', f'_Nx{C}x{H}x{W}')
        if not os.path.isdir(args.output): out_name = os.path.basename(args.output)
        output_onnx = os.path.join(os.path.dirname(args.output), out_name.replace('-','_') + '.onnx')

        torch.onnx.export(
            model=model,
            args=inputs,
            f=output_onnx,
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
            # dynamic_axes=dynamic_axes
        )
        model_onnx1 = onnx.load(output_onnx)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, output_onnx)
        model_onnx2 = onnx.load(output_onnx)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, output_onnx)
        model_onnx2 = onnx.load(output_onnx)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, output_onnx)
        model_onnx2 = onnx.load(output_onnx)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, output_onnx)

        initialize(
            input_onnx_file_path=output_onnx,
            output_onnx_file_path=output_onnx,
            initialization_character_string='N',
        )

        print(f">>> Saved at: {os.path.abspath(output_onnx)}")
        print('=' * 80)
        print()
