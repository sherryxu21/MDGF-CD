import torch
import torch.nn as nn


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


def _to_channel_last(x):
    return x.permute(0, 2, 3, 1)

def to_shape(x, input_resolution):
    H, W = input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    x = x.view(B, H, W, C)
    x = x.permute(0, 3, 1, 2) #[B,C,H,W]
    return x

def to_the_shape(x):
    B, L, C = x.shape
    H = int(L ** 0.5)
    W = H if L % H == 0 else L // H
    x = x.view(B, H, W, C)
    x = x.permute(0, 3, 1, 2)  # [B,C,H,W]
    return x


class SE(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeatExtract(nn.Module):

    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class GlobalQueryGen(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 image_resolution,
                 window_size,
                 num_heads):
        super().__init__()
        if input_resolution == image_resolution//4:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//8:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//16:

            if window_size == input_resolution:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )

            else:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=False)
                )

        elif input_resolution == image_resolution//32:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )

        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')

    def forward(self, x):
        x = _to_channel_last(self.to_q_global(x))
        B = x.shape[0]
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x