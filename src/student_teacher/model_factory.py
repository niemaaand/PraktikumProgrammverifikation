import torch
from torch import nn

from src.student_teacher.options import OptionsInformation
from src.student_teacher.student_models.mnist_fc_256_128_10 import MnistFc_784_256_256_10


def load_model(options: OptionsInformation) -> nn.Module:
    model = build_model(options)
    model.load_state_dict(torch.load(options.reload_model_path, map_location=torch.device(options.device)))
    return model


def build_model(options: OptionsInformation) -> nn.Module:
    if options.student_model == "mnist_fc_256_128_10":
        return MnistFc_784_256_256_10(options)
    else:
        raise NotImplementedError

