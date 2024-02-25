from src.student_teacher import model_factory
from src.student_teacher.options import OptionsInformation
from src.utils.onnx_utils import export_model_to_onnx
import torch


def build_debug_network(onnx_path: str, options: OptionsInformation):

    model = model_factory.build_model(options)
    model.set_weights()

    input = torch.Tensor(2)
    input[0] = 1
    input[1] = 1

    export_model_to_onnx(model, onnx_path)

    pass

