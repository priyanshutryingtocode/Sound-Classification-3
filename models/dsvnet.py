import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.30):
        super().__init__()

        # Main branch
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut branch
        self.shortcut = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            bias=True
        )

        self.dropout1 = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):

        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut branch
        identity = self.shortcut(x)

        # Residual addition
        out = out + identity

        # Regularization and downsampling
        out = self.dropout1(out)
        out = self.pool(out)
        out = self.dropout2(out)

        return out
    
class DSVNet(nn.Module):
    """
    DeepSoundVisionNet (DSVNet)

    Paper architecture:

        Input       224 x 224 x 3
        Conv        64
        Pool        112 x 112

        Residual    128
        Pool        56 x 56

        Residual    256
        Pool        28 x 28

        Residual    512
        Pool        14 x 14

        Flatten     100352
        Dense       1024
        Dropout     0.5
        Dense       10
    """

    def __init__(
        self,
        num_classes=10,
        dropout=0.30
    ):
        super().__init__()

        # --------------------------------------------------
        # Initial layer
        # --------------------------------------------------

        self.initial_conv = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.initial_bn = nn.BatchNorm2d(
            64
        )

        self.initial_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.initial_dropout = nn.Dropout(
            p=dropout
        )

        # --------------------------------------------------
        # Residual stages
        # --------------------------------------------------

        self.block1 = ResidualBlock(
            64,
            128,
            dropout
        )

        self.block2 = ResidualBlock(
            128,
            256,
            dropout
        )

        self.block3 = ResidualBlock(
            256,
            512,
            dropout
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            14 * 14 * 512,
            1024,
            bias=True
        )

        self.fc_dropout = nn.Dropout(
            p=0.50
        )

        self.fc2 = nn.Linear(
            1024,
            num_classes,
            bias=True
        )

    def forward(self, x):

        # Initial block
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_pool(x)
        x = self.initial_dropout(x)

        # Residual stages
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x