import torch
from torch import nn

from src.student_teacher.options import OptionsInformation
from src.student_teacher.student_models.acasxu import AcasXU0
from src.student_teacher.student_models.mini_debug_model import MiniDebugModel, MiniDebugModel2, MiniDebugModel2BIG
from src.student_teacher.student_models.mnist_fc import MnistFc_784_256_256_10, MnistFc_784_256_10, \
    MnistFc_784_256_256_256_10, MnistFc_784_64_10, MnistFc_784_32_10


def load_model(options: OptionsInformation) -> nn.Module:
    model = build_model(options)
    model.load_state_dict(torch.load(options.reload_model_path, map_location=torch.device(options.device)))
    return model


def build_model(options: OptionsInformation) -> nn.Module:

    if options.student_model == "mnist_fc_784_32_10":
        return MnistFc_784_32_10(options)
    elif options.student_model == "mnist_fc_784_64_10":
        return MnistFc_784_64_10(options)
    elif options.student_model == "minst_fc_784_256_10":
        return MnistFc_784_256_10(options)
    elif options.student_model == "mnist_fc_784_256_256_10":
        return MnistFc_784_256_256_10(options)
    elif options.student_model == "mnist_fc_784_256_256_256_10":
        return MnistFc_784_256_256_256_10(options)
    elif options.student_model == "acasxu_0":
        return AcasXU0(options)
    elif options.student_model == "mini_debug_model":
        return MiniDebugModel(options)
    elif options.student_model == "mini_debug_model2":
        return MiniDebugModel2(options)
    elif options.student_model == "mini_debug_model2_BIG":
        return MiniDebugModel2BIG(options)
    else:
        raise NotImplementedError

