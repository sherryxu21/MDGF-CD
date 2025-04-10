import torch
from torch import nn
import torch.nn.functional as F



def add_conv(in_ch, out_ch, ksize, stride):
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual
        return x


class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape

        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)

        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        x = x_maxpool + x_avgpool
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = inputs * x

        return outputs


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        outputs = inputs * x

        return outputs


class FMM(nn.Module):
    def __init__(self, dim):
        super(FMM, self).__init__()
        self.dim = dim
        self.conv_a = nn.Conv2d(self.dim, self.dim//4, 1, 1)
        self.conv_b = nn.Conv2d(self.dim//4, self.dim, 1, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(self.dim, self.dim, 3, 1, 1),
                                   nn.BatchNorm2d(self.dim),
                                   nn.LeakyReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim, self.dim, 3, 1, 3, dilation=3),
                                   nn.BatchNorm2d(self.dim),
                                   nn.LeakyReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.dim, self.dim, 3, 1, 5, dilation=5),
                                   nn.BatchNorm2d(self.dim),
                                   nn.LeakyReLU(True))
        self.conv4 = add_conv(3 * self.dim, self.dim, 1, 1)
        self.conv_1 = nn.Conv2d(self.dim, 1, 1, 1)
        self.conv_2 = nn.Conv2d(self.dim, 1, 1, 1)
        self.conv_3 = add_conv(2 * self.dim, self.dim, 1, 1)
        self.nam = Channel_Att(self.dim)

    def forward_semantic(self, x):
        x1 = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = self.conv_a(x1)
        x1 = self.conv_b(x1)
        x1 = x1 * x
        x1 = F.softmax(x1, dim=1)
        return x1

    def forward_multiscale(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv4(x)
        return x

    def forward(self, x, y):
        x1 = self.forward_semantic(x)
        y1 = self.forward_multiscale(y)
        x2 = self.conv_1(x1)
        a1 = F.softmax(torch.matmul(x2, y1), dim=1)
        a1 = a1 + y
        y2 = self.conv_2(y1)
        a2 = F.softmax(torch.matmul(y2, x1), dim=1)
        a2 = a2 + x
        a = torch.cat((a1, a2), 1)
        a = self.conv_3(a)
        a = self.nam(a)
        return a

class HFFM(nn.Module):
    def __init__(self, dim = [768,384,192,96]):
        super(HFFM, self).__init__()
        self.mapping_0 = add_conv(dim[0], dim[3], 3, 1)
        self.mapping_1 = add_conv(dim[1]+dim[3], dim[3], 3, 1)
        self.mapping_2 = add_conv(dim[2]+dim[3], dim[3], 3, 1)
        self.mapping_3 = add_conv(dim[3], dim[3], 3, 1)
        self.nam = Channel_Att(dim[3])

    def forward(self, x_level_0, x_level_1, x_level_2):
        fout_1 = self.mapping_0(x_level_0)   #96
        fout_1 = F.interpolate(fout_1, scale_factor=2, mode='bilinear', align_corners=False)
        fc_1 = torch.cat((fout_1, x_level_1), 1)   #384+96
        fout_2 = self.mapping_1(fc_1)  #96
        fout_2 = F.interpolate(fout_2, scale_factor=2, mode='bilinear', align_corners=False)
        fc_2 = torch.cat((fout_2, x_level_2), 1)   #192+96
        fout_3 = self.mapping_2(fc_2)   #96
        fout_3 = F.interpolate(fout_3, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.mapping_3(fout_3)
        out = self.nam(out)
        return out


class LFFM(nn.Module):
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        super(LFFM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                   nn.Conv2d(in_channel, in_channel, 1, 1),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU())

        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channel, in_channel, 1, 1)

    def forward(self, x, y, z):
        y = self.conv1(y)
        a = x + y + z
        a1 = self.channel_attention(a)
        a2 = self.spatial_attention(a)
        a = a1 + a2
        a = self.sigmoid(a)
        a = a * x + a * y
        out = a + x + y
        out = self.conv2(out)
        return out