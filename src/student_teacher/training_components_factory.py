import torch.optim
import torch.nn as nn

from src.student_teacher.options import OptionsInformation
from src.utils import nn_utils


def build_optimizer(options: OptionsInformation, model: nn.Module):
    params = model.parameters()

    if options.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.lr_momentum)
    if options.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=options.lr)
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(options: OptionsInformation, optimizer):
    if options.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=options.scheduler_step,
            gamma=options.scheduler_lr_gamma, last_epoch=-1)
    else:
        raise NotImplementedError

    return scheduler


def build_criterion(options: OptionsInformation, device=None):

    if options.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss().to(nn_utils.select_device(device))
    elif options.criterion == "MSELoss":
        criterion = nn.MSELoss().to(nn_utils.select_device(device))
    elif options.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().to(nn_utils.select_device(device))
    else:
        raise NotImplementedError

    return criterion

