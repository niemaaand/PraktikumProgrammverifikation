from torch import nn

from src.student_teacher.options import OptionsInformation


class MnistFc_784_256_128_10(nn.Module):
    def __init__(self, options: OptionsInformation):

        super(MnistFc_784_256_128_10, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256, device=options.device, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256, device=options.device, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10, device=options.device, bias=True)

        layers = [
            self.flatten, self.fc1, self.relu1
        ]

        for i in range (5):
            layers.append(nn.Linear(256, 256, device=options.device, bias=True))
            layers.append(nn.ReLU())

        layers.append(self.fc3)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

