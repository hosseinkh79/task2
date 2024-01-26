import torch
from torch import nn
from torch.nn import functional as F


class CRNN(nn.Module):
    def __init__(self, configs):
        super(CRNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(configs.INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Recurrent layers (LSTM)
        self.lstm1 = nn.LSTM(64 * (configs.IMAGE_HEIGHT // 4), 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)

        # Fully connected layer for output sequence
        self.fc_sequence = nn.Linear(256, configs.NUM_CLASSES +1)

    def forward(self, x):
        bs, _, _, _ = x.shape
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)

        # Recurrent layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.fc_sequence(x)
        x = x.permute(1, 0, 2)

        return x
    