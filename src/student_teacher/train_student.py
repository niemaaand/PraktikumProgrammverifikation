import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel

from src.student_teacher import model_factory
from src.student_teacher.options import OptionsInformation
from src.student_teacher.trainer import StudentTrainer


def load_teacher(path_to_onnx_model) -> nn.Module:
    onnx_model = onnx.load(path_to_onnx_model)
    pytorch_model = ConvertModel(onnx_model, experimental=True)
    return pytorch_model


def build_student(options: OptionsInformation) -> nn.Module:
    model = model_factory.build_model(options)
    return model


def train_student(student_model: nn.Module, teacher_model: nn.Module, options: OptionsInformation) -> nn.Module:
    trainer = StudentTrainer(student_model, teacher_model, options)
    trainer.train_model()
    trainer.test_model()

    # TODO: do not return but save models. Maybe load beste model to be returned.
    trained_student_model = trainer.student_model
    return trained_student_model


def save_student():
    pass


def create_student(teacher_model: nn.Module, options: OptionsInformation) -> nn.Module:
    student_model = build_student(options)
    student_model = train_student(student_model, teacher_model, options)
    save_student(student_model)