import torch
from torch import nn


def get_conv_output(shape, layer):
    bs = 1
    x = torch.autograd.Variable(torch.rand(bs, *shape))
    out = layer(x)
    return out.size()[1:]


class ShortNet(nn.Module):
    """Convolutional Neural Network short layers."""

    def __init__(self, input_shape, n_classes=2):
        """
        :param tuple input_shape: shape of inputs tensor.
            example: (3, 96, 96) with 3 for RGB.
        """
        super(ShortNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        shape = get_conv_output(input_shape, self.layer1)
        shape = get_conv_output(shape, self.layer2)

        self.fc1 = nn.Linear(shape[0] * shape[1] *
                             shape[2], 120)  # fully connected
        self.fc2 = nn.Linear(120, 84)  # fully connected
        self.fc3 = nn.Linear(84, n_classes)  # fully connected

    def forward(self, x, training=True):
        out = self.layer1(x)
        out = nn.functional.dropout2d(out, 0.15)
        out = self.layer2(out)
        out = nn.functional.dropout2d(out, 0.20)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
