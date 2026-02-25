import torch
import torch.nn as nn


class ConvBNActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.actfn = nn.SiLU()  # Swish activation function

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.actfn(x)
        return x



class Backbone(nn.Module):
    def __init__(self, c_in: int = 3, base: int = 32):
        super().__init__()
        self.stem = ConvBNActivation(c_in, base, kernel_size=3, stride=2)  # [N, c_in, 416, 416] -> [N, base, 208, 208]
        self.s1 = nn.Sequential(
            ConvBNActivation(base, base * 2, kernel_size=3, stride=2),
            ConvBNActivation(base * 2, base * 2, kernel_size=3),
        )  # [N, base, 208, 208] -> [N, 2*base, 104, 104]
        self.s2 = nn.Sequential(
            ConvBNActivation(base * 2, base * 4, kernel_size=3, stride=2),
            ConvBNActivation(base * 4, base * 4, kernel_size=3),
        )  # [N, 2*base, 104, 104] -> [N, 4*base, 52, 52]
        self.s3 = nn.Sequential(
            ConvBNActivation(base * 4, base * 8, kernel_size=3, stride=2),
            ConvBNActivation(base * 8, base * 8, kernel_size=3),
        )  # [N, 4*base, 52, 52] -> [N, 8*base, 26, 26]
        self.s4 = nn.Sequential(
            ConvBNActivation(base * 8, base * 16, kernel_size=3, stride=2),
            ConvBNActivation(base * 16, base * 16, kernel_size=3),
        )  # [N, 8*base, 26, 26] -> [N, 16*base, 13, 13]
    
    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        return x
    

class YOLOv1(nn.Module):
    def __init__(self, S: int = 13, C: int = 1, base: int = 32):
        super().__init__()
        self.S, self.C = S, C
        self.backbone = Backbone(base=base)  # [N, 3, 416, 416] -> [N, 16*base, 13, 13]
        D = base * 16
        out_ch = 5 + C
        self.head = nn.Conv2d(D, out_ch, kernel_size=1, stride=1, padding=0)  # [N, D, 13, 13] -> [N, 5+C, 13, 13]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)           # [N, D, 13, 13]
        p = self.head(f)               # [N, (5+C), 13, 13]
        p = p.permute(0, 2, 3, 1).contiguous()  # [N, 13, 13, (5+C)]
        return p
