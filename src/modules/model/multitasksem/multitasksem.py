import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Basic 2D building blocks
# ------------------------------
class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.same_channels = (in_ch == out_ch) and (stride == 1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if not self.same_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

class SqueezeExcitation2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BranchBlock2D(nn.Module):
    def __init__(self, channels, use_sem=True):
        super().__init__()
        self.res1 = ResBlock2D(channels, channels)
        self.sem = SqueezeExcitation2D(channels) if use_sem else nn.Identity()
        self.res2 = ResBlock2D(channels, channels)
    def forward(self, x):
        x = self.res1(x)
        x = self.sem(x)
        return self.res2(x)

# ------------------------------
# Multi-task 2D network
# ------------------------------
class MultiTaskSurvivalStageNet2D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, use_sem=True, dropout=0.4):
        super().__init__()
        # Shared encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shared1 = ResBlock2D(base_channels, base_channels*2, stride=1)
        self.shared2 = ResBlock2D(base_channels*2, base_channels*4, stride=2)
        self.shared3 = ResBlock2D(base_channels*4, base_channels*8, stride=2)

        # Survival branch
        self.surv_branch_block1 = BranchBlock2D(base_channels*8, use_sem=use_sem)
        self.surv_branch_block2 = BranchBlock2D(base_channels*8, use_sem=use_sem)
        self.surv_pool = nn.AdaptiveAvgPool2d(1)
        self.surv_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels*8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # risk score for Cox, or sigmoid for binary
        )

        # Stage branch
        self.stage_branch_block1 = BranchBlock2D(base_channels*8, use_sem=use_sem)
        self.stage_branch_block2 = BranchBlock2D(base_channels*8, use_sem=use_sem)
        self.stage_pool = nn.AdaptiveAvgPool2d(1)
        self.stage_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels*8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # binary stage logit
        )

    def forward(self, x):
        # Shared encoder
        x = self.stem(x)
        x = self.shared1(x)
        x = self.shared2(x)
        x = self.shared3(x)

        # Survival branch
        s = self.surv_branch_block1(x)
        s = self.surv_branch_block2(s)
        s = self.surv_pool(s)
        surv_risk = self.surv_mlp(s).squeeze(1)

        # Stage branch
        t = self.stage_branch_block1(x)
        t = self.stage_branch_block2(t)
        t = self.stage_pool(t)
        stage_logit = self.stage_mlp(t).squeeze(1)

        return surv_risk, stage_logit
