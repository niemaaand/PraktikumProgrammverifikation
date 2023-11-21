import copy
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchmetrics
import torchvision
import math

from src.student_teacher.options import OptionsInformation
from src.student_teacher import training_components_factory


class LossInformation:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.loss: float = 0.0
        self.accuracy: float = 0.0
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.f1: float = 0.0

    def get_comparison_score(self):
        return self.f1


class BestModel:
    def __init__(self, options_path, epoch=0, score=0):
        self.options_path = options_path
        self.epoch = epoch
        self.score = score


def build_model_name(score, epoch, options_file_name) -> str:
    model_name = "{}_f1Score__{}_Epoch__{}_Options.pt".format(score, epoch, options_file_name)
    model_name = model_name.replace(":", "_")
    return model_name


class StudentTrainer(object):
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, options: OptionsInformation, device=None):
        self.options = options
        self.device = options.device
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.optimizer = training_components_factory.build_optimizer(self.options, self.student_model)
        self.scheduler = training_components_factory.build_scheduler(self.options, self.optimizer)
        self.criterion = training_components_factory.build_criterion(self.options, self.device)
        self.student_training_loss_information = LossInformation("Student (Training data)")
        self.student_test_loss_information = LossInformation("Student (Test data)")
        self.teacher_loss_information = LossInformation("Teacher (Test data)")
        self.current_epoch: int = 0
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # TODO: verify number of classes
        self.highest_f1_score = 0.0

        self.save_directory = os.path.abspath("saved_models")

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.options_file_path = os.path.join(
            self.save_directory,
            "{}_options.json".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        self.options.write_options_to_json(self.options_file_path)

        self.best_model = BestModel(self.options_file_path)

        average_type_in_metrics = "macro"

        self.precision = torchmetrics.classification.MulticlassPrecision(len(self.classes),
                                                                         average=average_type_in_metrics,
                                                                         validate_args=False).to(self.device)
        self.recall = torchmetrics.classification.MulticlassRecall(len(self.classes), average=average_type_in_metrics,
                                                                   validate_args=False).to(self.device)
        self.f1 = torchmetrics.classification.MulticlassF1Score(len(self.classes), average=average_type_in_metrics,
                                                                validate_args=False).to(self.device)
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(len(self.classes),
                                                                       average=average_type_in_metrics,
                                                                       validate_args=False).to(self.device)

    def update_best_model(self, loss_info: LossInformation):
        if self.best_model.score < loss_info.get_comparison_score():
            self.best_model.score = loss_info.get_comparison_score()
            self.best_model.epoch = self.current_epoch
        # else: do nothing

    def train_model(self):
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()  # ,
                                           # torchvision.transforms.Normalize(
                                           #    (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.options.batch_size, shuffle=True)

        for self.current_epoch in range(self.options.n_epochs):

            self.student_training_loss_information.reset()

            self.student_model.train()
            self.teacher_model.eval()

            student_predict_total = torch.Tensor().to(self.device)
            teacher_predict_total = torch.Tensor().to(self.device)
            labels_total = torch.Tensor().to(self.device)

            for batch_idx, (pictures, labels) in enumerate(train_loader):
                pictures = pictures.to(self.device)

                teacher_logits = self.teacher_model(pictures)
                student_logits = self.student_model(pictures)

                self.optimizer.zero_grad()

                loss = self.criterion(student_logits, teacher_logits)
                loss.backward()
                self.optimizer.step()

                self.student_training_loss_information.loss += loss.item() / len(train_loader)

                _, student_predict = torch.max(student_logits.data, 1)
                _, teacher_predict = torch.max(teacher_logits.data, 1)
                student_predict_total = torch.cat((student_predict_total, student_predict))
                teacher_predict_total = torch.cat((teacher_predict_total, teacher_predict))
                labels_total = torch.cat((labels_total, labels))

            self.scheduler.step()
            self.student_training_loss_information = self.calc_metrics(self.student_training_loss_information,
                                                                       student_predict_total, labels_total)
            self.print_info(self.student_training_loss_information, self.current_epoch)
            self.test_model(print_teacher=False)
            self.update_best_model(self.student_test_loss_information)
            self.save_student(self.student_test_loss_information, self.save_directory)

    def print_info(self, loss: LossInformation, epoch: int):
        info: str = "{} | Epoch: {} | Loss: {} | Accuracy: {} | Precision: {} | Recall: {} | F1: {}".format(
            loss.name, epoch, loss.loss, loss.accuracy, loss.precision, loss.recall, loss.f1)

        print(info)

    def test_model(self, print_teacher: bool = True):
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()  # ,
                                           # torchvision.transforms.Normalize(
                                           # (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.options.batch_size, shuffle=True)

        self.student_test_loss_information.reset()
        self.teacher_loss_information.reset()

        self.student_model.eval()
        self.teacher_model.eval()

        student_predict_total = torch.Tensor().to(self.device)
        teacher_predict_total = torch.Tensor().to(self.device)
        labels_total = torch.Tensor().to(self.device)

        for batch_idx, (pictures, labels) in enumerate(test_loader):

            pictures = pictures.to(self.device)

            student_logits = self.student_model(pictures)
            student_loss = self.criterion(student_logits, labels)

            self.student_test_loss_information.loss += student_loss.item() / len(test_loader)

            _, student_predict = torch.max(student_logits.data, 1)
            student_predict_total = torch.cat((student_predict_total, student_predict))
            labels_total = torch.cat((labels_total, labels))

            if print_teacher:
                teacher_logits = self.teacher_model(pictures)
                teacher_loss = self.criterion(teacher_logits, labels)
                self.teacher_loss_information.loss += teacher_loss.item() / len(test_loader)
                _, teacher_predict = torch.max(teacher_logits.data, 1)
                teacher_predict_total = torch.cat((teacher_predict_total, teacher_predict))

        self.student_test_loss_information = self.calc_metrics(self.student_test_loss_information,
                                                               student_predict_total, labels_total)
        self.print_info(self.student_test_loss_information, self.current_epoch)

        if print_teacher:
            self.teacher_loss_information = self.calc_metrics(self.teacher_loss_information, teacher_predict_total,
                                                              labels_total)
            self.print_info(self.teacher_loss_information, self.current_epoch)

    def calc_metrics(self, loss: LossInformation, predict_total, labels_total) -> LossInformation:
        loss.accuracy = self.accuracy(predict_total, labels_total)
        loss.precision = self.precision(predict_total, labels_total)
        loss.recall = self.recall(predict_total, labels_total)
        loss.f1 = self.f1(predict_total, labels_total)

        return loss

    def save_student(self, loss_info: LossInformation, save_directory, threshold=0.8):
        score = loss_info.get_comparison_score()
        epoch = self.current_epoch
        options_file_name = os.path.basename(self.options_file_path)

        if score > threshold:
            model_name = build_model_name(score, epoch, options_file_name)
            torch.save(self.student_model.state_dict(), os.path.join(save_directory, model_name))
        # else: do nothing

    def get_best_student(self) -> (nn.Module, str):
        if self.best_model and self.best_model.score:
            best_student_path = build_model_name(self.best_model.score, self.best_model.epoch,
                                                 os.path.basename(self.best_model.options_path))
        elif self.options.reload_model_path:
            best_student_path = self.options.reload_model_path
        else:
            print("No model was good enough.")

        best_student_path = os.path.join(self.save_directory, best_student_path)

        if os.path.exists(best_student_path):
            student_model = copy.deepcopy(self.student_model)

            student_model.load_state_dict(
                torch.load(best_student_path, map_location=self.options.device))

            return student_model, best_student_path

        return None, None
