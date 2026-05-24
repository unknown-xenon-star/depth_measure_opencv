import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------
# Depthwise Separable Convolution
# -----------------------------------
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# -----------------------------------
# Learning To Downsample
# -----------------------------------
class LearningToDownsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DSConv(32, 48, stride=2),
            DSConv(48, 64, stride=2)
        )

    def forward(self, x):
        return self.layers(x)


# -----------------------------------
# Global Feature Extractor
# -----------------------------------
class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            DSConv(64, 96, stride=2),
            DSConv(96, 128, stride=2),
            DSConv(128, 128, stride=1)
        )

    def forward(self, x):
        return self.layers(x)


# -----------------------------------
# Feature Fusion
# -----------------------------------
class FeatureFusion(nn.Module):
    def __init__(self, high_ch, low_ch, out_ch):
        super().__init__()

        self.dwconv = nn.Conv2d(
            high_ch,
            out_ch,
            3,
            padding=1,
            groups=high_ch,
            bias=False
        )

        self.conv_low = nn.Conv2d(low_ch, out_ch, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_res, low_res):

        low_res = F.interpolate(
            low_res,
            size=high_res.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        high = self.dwconv(high_res)
        low = self.conv_low(low_res)

        out = high + low
        out = self.bn(out)

        return self.relu(out)


# -----------------------------------
# Classifier
# -----------------------------------
class Classifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            DSConv(in_ch, in_ch),
            DSConv(in_ch, in_ch),
            nn.Conv2d(in_ch, num_classes, 1)
        )

    def forward(self, x):
        return self.layers(x)


# -----------------------------------
# Fast-SCNN
# -----------------------------------
class FastSCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.downsample = LearningToDownsample()

        self.global_features = GlobalFeatureExtractor()

        self.fusion = FeatureFusion(
            high_ch=64,
            low_ch=128,
            out_ch=128
        )

        self.classifier = Classifier(
            in_ch=128,
            num_classes=num_classes
        )

    def forward(self, x):

        input_size = x.shape[2:]

        high_res = self.downsample(x)

        low_res = self.global_features(high_res)

        fused = self.fusion(high_res, low_res)

        out = self.classifier(fused)

        out = F.interpolate(
            out,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )

        return out


# -----------------------------------
# Test
# -----------------------------------
if __name__ == "__main__":

    model = FastSCNN(num_classes=19)

    x = torch.randn(1, 3, 512, 1024)

    y = model(x)

    print("Output shape:", y.shape)