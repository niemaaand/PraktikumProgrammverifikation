import copy
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchmetrics
import torchvision

from src.student_teacher.datasets import RandomDataDataset
from src.student_teacher.options import OptionsInformation, DatasetType
from src.student_teacher import training_components_factory
from src.vnnlib.properties import Properties


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
    model_name = "{}_Options__{}_f1Score__{}_Epoch.pt".format(options_file_name, score, epoch)
    model_name = model_name.replace(":", "_")
    return model_name


class StudentTrainer(object):
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, options: OptionsInformation, device=None,
                 vnnlibs_path=""):
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

        self._T = self.options.T if self.options.T else 1

        self.save_directory = os.path.abspath("saved_models")

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.options_file_path = os.path.join(
            self.save_directory,
            "{}_options.json".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        self.options.write_options_to_json(self.options_file_path)

        self.properties = Properties(vnnlibs_path, self._calc_in_features(), self._calc_out_features())

        self.best_model = BestModel(self.options_file_path)

        average_type_in_metrics = self.options.average_type_in_metrics

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

        self.test_loader = self.get_data_loader(self._calc_datatuple_sizes()[1:], train=False,
                                                n_samples=self.options.n_batches_validation * self.options.batch_size)

    def update_best_model(self, loss_info: LossInformation):
        if self.best_model.score < loss_info.get_comparison_score():
            self.best_model.score = loss_info.get_comparison_score()
            self.best_model.epoch = self.current_epoch
        # else: do nothing

    def get_data_loader(self, size, train, n_samples=1):

        if self.options.dataset == DatasetType.MNIST:
            return torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('/files/', train=train, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor()  # ,
                                               # torchvision.transforms.Normalize(
                                               #    (0.1307,), (0.3081,))
                                           ])),
                batch_size=self.options.batch_size, shuffle=True)
        elif self.options.dataset == DatasetType.RANDOM_INPUTS:
            if train:
                return None
            else:
                return torch.utils.data.DataLoader(
                    RandomDataDataset(size, n_samples, self.properties), batch_size=self.options.batch_size,
                    shuffle=True, drop_last=False
                )

    def _calc_in_features(self):

        try:
            return self.student_model.get_input_size()
        except:
            print("Student model missing function: get_input_size()")

        in_features = 0
        for layer in self.student_model.layers:
            if hasattr(layer, "in_features"):
                in_features = layer.in_features
                break

        return in_features

    def _calc_out_features(self):

        try:
            return self.student_model.get_output_size()
        except:
            print("Student model missing function: get_output_size()")


        out_features = 0
        cnt = len(self.student_model.layers) - 1
        while cnt >= 0:
            if hasattr(self.student_model.layers[cnt], "out_features"):
                out_features = self.student_model.layers[cnt].out_features
                break

        return out_features

    def _calc_datatuple_sizes(self):
        in_features = self._calc_in_features()

        size = [self.options.batch_size, 1]
        try:
            for f in in_features:
                size.append(f)
        except TypeError:
            size.append(in_features)

        return size

    def get_next_data_tuple(self, data_loader_iterator, batch_idx, n_batches):

        if self.options.dataset == DatasetType.MNIST and data_loader_iterator:
            try:
                pictures, labels = data_loader_iterator.__next__()
                return batch_idx + 1, (pictures, labels)
            except StopIteration:
                pass
        elif self.options.dataset == DatasetType.RANDOM_INPUTS:
            if not batch_idx >= n_batches:
                size = self._calc_datatuple_sizes()

                rand_vec = self.properties.calc_random_input_vector(size)

                return batch_idx + 1, (rand_vec, None)
                # return batch_idx + 1, (torch.rand(size), None)
        else:
            raise NotImplementedError

        return batch_idx + 1, (None, None)

    def calc_loss(self, student_logits, teacher_logits):
        # https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html

        label_loss = self.criterion(student_logits, teacher_logits)

        #criterion2 = nn.BCEWithLogitsLoss()
        #label_loss2 = criterion2(student_logits, teacher_logits)

        # scale Temperature down to 1
        if not self._T or self.current_epoch % self.options.adapt_T_steps == 0 and self.options.adapt_T:
            self._T = self.options.T if not self.options.adapt_T else (self.options.T - 1 ) - ((self.options.T-1) / (self.options.n_epochs - 1)) * self.current_epoch + 1

        # Soften the student logits by applying softmax first and log() second
        if student_logits.size() != teacher_logits.size():
            teacher_logits_not_labels = torch.eye(student_logits.size()[-1])[teacher_logits]
            assert student_logits.size() == teacher_logits_not_labels.size()
        else:
            teacher_logits_not_labels = teacher_logits
        soft_targets = nn.functional.softmax(teacher_logits_not_labels / self._T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self._T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (self._T ** 2)

        #loss = soft_target_loss_weight * soft_targets_loss + (ce_loss_weight / 2) * label_loss + label_loss2 * (ce_loss_weight / 2)

        loss = self.options.soft_target_loss_weight * soft_targets_loss + self.options.ce_loss_weight * label_loss
        return loss

    def train_model(self):

        train_loader = self.get_data_loader(self._calc_datatuple_sizes(), train=True)

        if train_loader:
            n_batches = len(train_loader)
        elif self.options.dataset == DatasetType.RANDOM_INPUTS:
            n_batches = self.options.n_batches

        for self.current_epoch in range(self.options.n_epochs):

            self.student_training_loss_information.reset()

            self.student_model.train()
            student_predict_total = torch.Tensor().to(self.device)
            labels_total = torch.Tensor().to(self.device)

            if self.teacher_model:
                self.teacher_model.eval()
                teacher_predict_total = torch.Tensor().to(self.device)

            train_loader_iterator = None
            if train_loader:
                train_loader_iterator = train_loader.__iter__()

            batch_idx, (pictures, labels) = self.get_next_data_tuple(train_loader_iterator, -1, self.options.n_batches)
            while (pictures != None):
                # if train_loader:
                # for batch_idx, (pictures, labels) in enumerate(train_loader):
                pictures = pictures.to(self.device)
                self.optimizer.zero_grad()

                student_logits = self.student_model(pictures)

                if self.teacher_model:
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(pictures)

                    loss = self.calc_loss(student_logits, teacher_logits)
                else:
                    loss = self.calc_loss(student_logits, labels)

                loss.backward()
                self.optimizer.step()

                self.student_training_loss_information.loss += loss.item() / n_batches

                _, student_predict = torch.max(student_logits.data, 1)

                if self.teacher_model:
                    _, teacher_predict = torch.max(teacher_logits.data, 1)
                    teacher_predict_total = torch.cat((teacher_predict_total, teacher_predict))

                student_predict_total = torch.cat((student_predict_total, student_predict))
                labels_total = torch.cat((labels_total, labels))

                batch_idx, (pictures, labels) = self.get_next_data_tuple(train_loader_iterator, batch_idx,
                                                                         self.options.n_batches)

            self.scheduler.step()
            if self.teacher_model:
                self.student_training_loss_information = self.calc_metrics(self.student_training_loss_information,
                                                                           student_predict_total, teacher_predict_total)
            else:
                self.student_training_loss_information = self.calc_metrics(self.student_training_loss_information,
                                                                           student_predict_total, labels_total)

            self.print_info(self.student_training_loss_information, self.current_epoch)
            self.test_model(print_teacher=bool(self.teacher_model))
            self.update_best_model(self.student_test_loss_information)
            self.save_student(self.student_test_loss_information, self.save_directory)

    def print_info(self, loss: LossInformation, epoch: int):
        info: str = "{} | Epoch: {} | Loss: {} | Accuracy: {} | Precision: {} | Recall: {} | F1: {}".format(
            loss.name, epoch, loss.loss, loss.accuracy, loss.precision, loss.recall, loss.f1)

        print(info)

    def test_model(self, print_teacher: bool = True):

        if self.options.dataset == DatasetType.RANDOM_INPUTS and not self.options.choose_best_model_based_on_teacher:
            self.options.dataset = DatasetType.MNIST
            self.test_loader = self.get_data_loader(self._calc_datatuple_sizes()[1:], train=False,
                                                    n_samples=self.options.n_batches_validation * self.options.batch_size)
            self.options.dataset = DatasetType.RANDOM_INPUTS

        if self.test_loader:
            test_loader = self.test_loader
        else:
            test_loader = self.get_data_loader([self.options.n_batches_validation] + self._calc_datatuple_sizes(),
                                               train=False)

        self.student_test_loss_information.reset()
        self.teacher_loss_information.reset()

        self.student_model.eval()

        if self.teacher_model:
            self.teacher_model.eval()

        student_predict_total = torch.Tensor().to(self.device)
        teacher_predict_total = torch.Tensor().to(self.device)
        labels_total = torch.Tensor().to(self.device)

        for batch_idx, (pictures, labels) in enumerate(test_loader):
            pictures = pictures.to(self.device)

            with torch.no_grad():
                student_logits = self.student_model(pictures)

                if self.teacher_model:
                    teacher_logits = self.teacher_model(pictures)
                    # teacher_loss = self.criterion(teacher_logits, labels)
                    # self.teacher_loss_information.loss += teacher_loss.item() / len(test_loader)
                    _, teacher_predict = torch.max(teacher_logits.data, 1)
                    teacher_predict_total = torch.cat((teacher_predict_total, teacher_predict))

                    student_loss = self.calc_loss(student_logits, teacher_logits)  # self.criterion(student_logits, teacher_logits)
                else:
                    student_loss = self.calc_loss(student_logits, labels)

                self.student_test_loss_information.loss += student_loss.item() / len(test_loader)

                _, student_predict = torch.max(student_logits.data, 1)
                student_predict_total = torch.cat((student_predict_total, student_predict))
                labels_total = torch.cat((labels_total, labels))

        ground_truth = teacher_predict_total if self.options.choose_best_model_based_on_teacher or self.options.dataset == DatasetType.RANDOM_INPUTS else labels_total
        self.student_test_loss_information = self.calc_metrics(self.student_test_loss_information,
                                                               student_predict_total, ground_truth)
        self.print_info(self.student_test_loss_information, self.current_epoch)

        if print_teacher and not self.options.dataset == DatasetType.RANDOM_INPUTS:
            self.teacher_loss_information.loss = "not calculated"
            self.teacher_loss_information = self.calc_metrics(self.teacher_loss_information, teacher_predict_total,
                                                             labels_total)
            self.print_info(self.teacher_loss_information, self.current_epoch)

    def calc_metrics(self, loss: LossInformation, predict_total, labels_total) -> LossInformation:
        loss.accuracy = self.accuracy(predict_total, labels_total)
        loss.precision = self.precision(predict_total, labels_total)
        loss.recall = self.recall(predict_total, labels_total)
        loss.f1 = self.f1(predict_total, labels_total)

        return loss

    def save_student(self, loss_info: LossInformation, save_directory):
        score = loss_info.get_comparison_score()
        epoch = self.current_epoch
        options_file_name = os.path.basename(self.options_file_path)

        if score > self.options.saving_threshold or self.options.force_save:
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
