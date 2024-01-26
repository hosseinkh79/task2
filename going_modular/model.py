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

        # self.convolutional_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=(4, 4))


        
        self.lstm1 = nn.LSTM(64 * (configs.IMAGE_HEIGHT // 4), 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)

        # Fully connected layer for output sequence
        self.fc_sequence = nn.Linear(256, configs.NUM_CLASSES +1)

    def forward(self, x):
        # x = F.relu(self.convolutional_layer(x))
        # bs, _, _, _ = x.shape


        # print(f'my conv size : {x.shape}')
        bs, _, _, _ = x.shape
        # Convolutional layers
        # print(f'x size : {x.shape}')
        x = F.relu(self.conv1(x))
        # print(f'x size : {x.shape}')
        x = self.pool1(x)
        # print(f'x size : {x.shape}')
        x = F.relu(self.conv2(x))
        # print(f'x size : {x.shape}')
        x = self.pool2(x)
        # print(f'final conv size : {x.shape}')
        x = x.permute(0, 3, 1, 2)
        # print(f'x size permute(0, 3, 1, 2) : {x.shape}')
        x = x.view(bs, x.size(1), -1)
        # print(f'x size . input to lstm : {x.shape}')

        # Recurrent layers
        x, _ = self.lstm1(x)
        # print(f'x size out lstm1 : {x.shape}')
        x, _ = self.lstm2(x)
        # print(f'x size after lstm : {x.shape}')

        x = self.fc_sequence(x)
        # print(f'x size : {x.shape}')
        x = x.permute(1, 0, 2)
        # print(f'x size : {x.shape}')
        # print(f'Finishhh--------------------------------------------------------')

        return x
    