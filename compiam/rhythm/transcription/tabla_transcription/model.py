try:
    import torch
    import torch.nn as nn
except:
    raise ImportError(
        "In order to use this tool you need to have torch installed. "
        "Please install torch using: pip install torch"
    )


class onsetCNN(nn.Module):
    """Model definition for resonant bass and resonant both categories
    """

    def __init__(self):
        super(onsetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 7))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((3, 1))
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((3, 1))
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(32 * 7 * 8, 128)
        self.bn4 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(128, 1)
        self.dout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.pool1(y)
        y = torch.relu(self.bn2(self.conv2(y)))
        y = self.pool2(y)
        y = self.dout2(y.view(-1, 32 * 7 * 8))
        y = self.dout2(torch.relu(self.bn3(self.fc1(y))))
        y = torch.sigmoid(self.bn4(self.fc2(y)))
        return y


class onsetCNN_D(nn.Module):
    """Model definition for damped category
    """

    def __init__(self):
        super(onsetCNN_D, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 7))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((3, 1))
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((3, 1))
        self.bn3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(32 * 7 * 8, 256)
        self.bn4 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(256, 1)
        self.dout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.pool1(y)
        y = torch.relu(self.bn2(self.conv2(y)))
        y = self.pool2(y)
        y = self.dout2(y.view(-1, 32 * 7 * 8))
        y = self.dout2(torch.relu(self.bn3(self.fc1(y))))
        y = torch.sigmoid(self.bn4(self.fc2(y)))
        return y


class onsetCNN_RT(nn.Module):
    """Model definition for resonant treble category
    """

    def __init__(self):
        super(onsetCNN_RT, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 7))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((3, 1))
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((3, 1))
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(64 * 7 * 8, 128)
        self.bn4 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(128, 1)
        self.dout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.pool1(y)
        y = torch.relu(self.bn2(self.conv2(y)))
        y = self.pool2(y)
        y = self.dout2(y.view(-1, 64 * 7 * 8))
        y = self.dout2(torch.relu(self.bn3(self.fc1(y))))
        y = torch.sigmoid(self.bn4(self.fc2(y)))
        return y
