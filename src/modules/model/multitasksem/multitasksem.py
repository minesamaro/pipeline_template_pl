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
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        
        # Squeeze: global average pooling → (B, C, 1, 1) → (B, C)
        y = self.global_pool(x).view(b, c)
        
        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        # Scale: multiply input by channel-wise weights
        scaled = x * y
        
        # Residual addition: scaled features + original features
        out = scaled + x
        
        return out


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
    def __init__(self, in_channels=1, base_channels=32, stage_classes=7, use_sem=True):
        super().__init__()
        # Shared encoder — no stem, directly ResBlocks
        self.shared1 = ResBlock2D(in_channels, base_channels, stride=1)
        self.shared2 = ResBlock2D(base_channels, base_channels * 2, stride=2)
        self.shared3 = ResBlock2D(base_channels * 2, base_channels * 4, stride=2)
        #self.shared4 = ResBlock2D(base_channels * 4, base_channels * 8, stride=2)

        # Survival branch
        self.surv_branch_block1 = BranchBlock2D(base_channels * 4, use_sem=use_sem)
        self.surv_branch_block2 = BranchBlock2D(base_channels * 4, use_sem=use_sem)
        self.surv_pool = nn.AdaptiveAvgPool2d(1)
        self.surv_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        # Stage branch
        self.stage_branch_block1 = BranchBlock2D(base_channels * 4, use_sem=use_sem)
        self.stage_branch_block2 = BranchBlock2D(base_channels * 4, use_sem=use_sem)
<<<<<<< HEAD
        self.stage_pool = nn.AvgPool2d(1)
=======
        self.stage_pool = nn.AdaptiveAvgPool2d(1)
>>>>>>> f0fd0dbef581bd3dfa43eb905be602e1459939e2
        self.stage_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, stage_classes)  # Multiclass output
        )

    def forward(self, x):
        # Shared encoder
        x = self.shared1(x)
        x = self.shared2(x)
        x = self.shared3(x)
        #x = self.shared4(x)

        # Survival branch
        s = self.surv_branch_block1(x)
        s = self.surv_branch_block2(s)
        s = self.surv_pool(s)
        surv_out = self.surv_mlp(s).squeeze(-1)

        # Stage branch
        t = self.stage_branch_block1(x)
        t = self.stage_branch_block2(t)
        t = self.stage_pool(t)
        stage_out = self.stage_mlp(t)  # logits for multiclass

        return surv_out, stage_out