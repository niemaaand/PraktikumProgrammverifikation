import torch
from torch import nn

from src.student_teacher.options import OptionsInformation


class MiniDebugModel(nn.Module):
    def __init__(self, options: OptionsInformation):

        super(MiniDebugModel, self).__init__()

        layers = [
            nn.Linear(2, 2, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(2, 1, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

    def set_weights(self):
        with torch.no_grad():
            self.layers[0].weight[0, 0] = 1
            self.layers[0].weight[0, 1] = 1
            self.layers[0].weight[1, 0] = 1
            self.layers[0].weight[1, 1] = -3

            self.layers[0].bias[0] = 1
            self.layers[0].bias[1] = 1

            self.layers[2].weight[0, 0] = 1
            self.layers[2].weight[0, 1] = 1

            self.layers[2].bias[0] = 1

        pass

    def get_random_input_sample(self):
        return torch.randn(2)


class MiniDebugModel2(nn.Module):
    def __init__(self, options: OptionsInformation):

        super(MiniDebugModel2, self).__init__()

        layers = [
            nn.Linear(2, 3, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(3, 1, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

    def set_weights(self):
        with torch.no_grad():
            self.layers[0].weight[0, 0] = 1
            self.layers[0].weight[0, 1] = 1
            self.layers[0].weight[1, 0] = 1
            self.layers[0].weight[1, 1] = -3
            self.layers[0].weight[2, 0] = -1
            self.layers[0].weight[2, 1] = 2

            self.layers[0].bias[0] = 1
            self.layers[0].bias[1] = 1
            self.layers[0].bias[2] = 0

            self.layers[2].weight[0, 0] = 1
            self.layers[2].weight[0, 1] = 1
            self.layers[2].weight[0, 2] = 2

            self.layers[2].bias[0] = 1

        pass

    def get_random_input_sample(self):
        return torch.randn(2)


class MiniDebugModel2BIG(nn.Module):
    def __init__(self, options: OptionsInformation):
        super(MiniDebugModel2BIG, self).__init__()

        layers = [
            nn.Linear(2, 4, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(4, 2, device=options.device, bias=True),
            nn.ReLU(),
            nn.Linear(2, 1, device=options.device, bias=True)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

    def set_weights(self):
        with torch.no_grad():
            self.layers[0].weight[0, 0] = 1
            self.layers[0].weight[0, 1] = 1
            self.layers[0].weight[1, 0] = 1
            self.layers[0].weight[1, 1] = -3
            self.layers[0].weight[2, 0] = -1
            self.layers[0].weight[2, 1] = 2
            self.layers[0].weight[3, 0] = 0
            self.layers[0].weight[3, 1] = 0

            self.layers[0].bias[0] = 1
            self.layers[0].bias[1] = 1
            self.layers[0].bias[2] = 0
            self.layers[0].bias[3] = 0

            self.layers[2].weight[0, 0] = 1
            self.layers[2].weight[0, 1] = 1
            self.layers[2].weight[0, 2] = 2
            self.layers[2].weight[0, 3] = 0
            self.layers[2].weight[1, 0] = 0
            self.layers[2].weight[1, 1] = 0
            self.layers[2].weight[1, 2] = 0
            self.layers[2].weight[1, 3] = 0

            self.layers[2].bias[0] = 1
            self.layers[2].bias[1] = 0

            self.layers[4].weight[0, 0] = 1
            self.layers[4].weight[0, 1] = 1

            self.layers[4].bias[0] = 1

        pass

    def get_random_input_sample(self):
        return torch.randn(2)


