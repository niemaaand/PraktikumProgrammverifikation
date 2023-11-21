from torch import nn

from src.student_teacher.options import OptionsInformation


class MnistFc_784_256_256_10(nn.Module):
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

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

