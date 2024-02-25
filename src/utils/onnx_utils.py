import os

import onnx
import torch
from torch import nn as nn


def export_model_to_onnx(model: nn.Module, file_path):

    file_path = os.path.abspath(file_path)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


    torch_input = model.get_random_input_sample()
    torch.onnx.export(model, torch_input, file_path)
    #onnx_program.save(file_path)

    onnx_model = onnx.load(file_path)
    onnx.checker.check_model(onnx_model)
