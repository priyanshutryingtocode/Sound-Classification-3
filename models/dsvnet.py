import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    DSVNet residual block.

    Main path:
        Conv 3x3 -> BatchNorm -> ReLU
        Conv 3x3 -> BatchNorm

    Shortcut:
        Conv 1x1 -> BatchNorm

    Both paths are added before ReLU.
    """

    def __init__(self, in_channels, out_channels, dropout=0.30):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.main(x)

        out = out + identity
        out = self.relu(out)

        out = self.dropout(out)
        out = self.pool(out)

        return out


class DSVNet(nn.Module):
    """
    DeepSoundVisionNet (DSVNet)

    Input:
        (batch_size, 3, 224, 224)

    Output:
        (batch_size, 10)
    """

    def __init__(self, num_classes=10, dropout=0.30):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )

        # Residual stages
        self.block1 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            dropout=dropout
        )

        self.block2 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            dropout=dropout
        )

        self.block3 = ResidualBlock(
            in_channels=256,
            out_channels=512,
            dropout=dropout
        )

        # The paper uses:
        # 224 -> 112 -> 56 -> 28 -> 14
        #
        # Therefore after block3:
        # feature map = 14 x 14 x 512
        #
        # Flattened size = 14 * 14 * 512 = 100352

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                14 * 14 * 512,
                1024
            ),
            nn.ReLU(inplace=True),

            nn.Dropout(0.50),

            nn.Linear(
                1024,
                num_classes
            )
        )

    def forward(self, x):
        x = self.initial(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.classifier(x)

        return x