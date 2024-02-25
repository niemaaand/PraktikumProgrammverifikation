import os.path

import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel

from src.student_teacher import model_factory
from src.student_teacher.options import OptionsInformation
from src.student_teacher.trainer import StudentTrainer
from src.utils.onnx_utils import export_model_to_onnx


def load_teacher(path_to_onnx_model) -> nn.Module:
    onnx_model = onnx.load(path_to_onnx_model)
    pytorch_model = ConvertModel(onnx_model, experimental=True)
    return pytorch_model


def build_student(options: OptionsInformation) -> nn.Module:
    model = model_factory.build_model(options)
    return model


def train_student(student_model: nn.Module, teacher_model: nn.Module, options: OptionsInformation) -> (nn.Module, str):
    trainer = StudentTrainer(student_model, teacher_model, options, vnnlibs_path=options.vnnlibs_path)
    trainer.train_model()
    best_student, best_student_path = trainer.get_best_student()

    return best_student, best_student_path


def load_model(options: OptionsInformation, state_dict_path: str):

    state_dict_path = os.path.abspath(state_dict_path)

    student_model = build_student(options)
    student_model.load_state_dict(
        torch.load(state_dict_path, map_location=options.device))
    onnx_path = os.path.abspath("onnx/{}.onnx".format(os.path.basename(os.path.splitext(state_dict_path)[0])))
    export_model_to_onnx(student_model, onnx_path)
    pass


def create_student(teacher_model: nn.Module, options: OptionsInformation) -> nn.Module:
    student_model = build_student(options)
    best_student_model, best_student_path = train_student(student_model, teacher_model, options)
    if best_student_path:
        onnx_path = os.path.abspath("onnx/{}.onnx".format(os.path.basename(os.path.splitext(best_student_path)[0])))
        export_model_to_onnx(best_student_model, onnx_path)
    else:
        print("Models to bad - no model saved.")

    pass
