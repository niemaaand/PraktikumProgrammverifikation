import torch
from torch import nn

from src.student_teacher.options import OptionsInformation


class AcasXU(nn.Module):
    def __init__(self):
        super(AcasXU, self).__init__()

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

    def get_random_input_sample(self):
        return torch.randn(1, 1, 5)

    def get_input_size(self):
        return 1, 5

    def get_output_size(self):
        return 5


class AcasXU0(AcasXU):
    def __init__(self, options: OptionsInformation):
        super(AcasXU0, self).__init__()

        layers = [
            nn.Linear(5, 30, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(30, 30, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(30, 30, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(30, 30, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(30, 5, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        out = super().forward(inputs)
        out = torch.flatten(out, start_dim=1)
        return out

