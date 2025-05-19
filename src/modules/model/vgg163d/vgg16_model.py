import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16_3DModel(nn.Module):
    def __init__(self, config):
        super(VGG16_3DModel, self).__init__()

        self.num_classes = config.number_of_classes
        self.in_channels = 3

        def conv_block(in_c, out_c, num_convs):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv3d(in_c, out_c, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_c = out_c
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(self.in_channels, 64, 2),   # Block 1
            conv_block(64, 128, 2),           # Block 2
            conv_block(128, 256, 3),          # Block 3
            conv_block(256, 512, 3),          # Block 4
            conv_block(512, 512, 3),          # Block 5
        )

        # Assuming input volumes are around (D=32, H=128, W=128)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1, 1)

        # Resize the input for B, C, 64, 512, 512
        x = self.features(x)           # (B, 512, D/32, H/32, W/32)
        x = self.avgpool(x)            # (B, 512, 1, 1, 1)
        x = torch.flatten(x, 1)        # (B, 512)
        x = self.classifier(x)         # (B, num_classes)
        return x
