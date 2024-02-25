import torch
from torch import nn

from src.student_teacher.options import OptionsInformation


class MnistFc(nn.Module):
    def __init__(self):
        super(MnistFc, self).__init__()

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

    def get_random_input_sample(self):
        return torch.randn(1, 1, 28, 28)

    def get_input_size(self):
        return (28, 28)

    def get_output_size(self):
        return 10



class MnistFc_784_32_10(MnistFc):
    def __init__(self, options: OptionsInformation):

        super(MnistFc_784_32_10, self).__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(784, 32, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(32, 10, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)



class MnistFc_784_64_10(MnistFc):
    def __init__(self, options: OptionsInformation):

        super(MnistFc_784_64_10, self).__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(784, 64, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)


class MnistFc_784_256_10(MnistFc):
    def __init__(self, options: OptionsInformation):

        super(MnistFc_784_256_10, self).__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(784, 256, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)


class MnistFc_784_256_256_10(MnistFc):
    def __init__(self, options: OptionsInformation):

        super(MnistFc_784_256_256_10, self).__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(784, 256, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)


class MnistFc_784_256_256_256_10(MnistFc):
    def __init__(self, options: OptionsInformation):

        super(MnistFc_784_256_256_256_10, self).__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(784, 256, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)

