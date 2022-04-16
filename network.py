# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2022 04 01 
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# 基本卷积模块
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect')
        self.relu = nn.ReLU(True)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = self.relu(out)
        return out


# 密集卷积的子模块
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# 密集卷积模块
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       CAMB(in_channels + out_channels_def),
                       DenseConv2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride),
                       CAMB(in_channels + out_channels_def * 2),
                       DenseConv2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride),
                       CAMB(in_channels + out_channels_def * 3)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# 编码器
class Dense_Encoder(nn.Module):
    def __init__(self, input_nc=1, kernel_size=3, stride=1):
        super(Dense_Encoder, self).__init__()
        self.conv = ConvLayer(input_nc, 16, kernel_size, stride)
        self.DenseBlock = DenseBlock(16, kernel_size, stride)

    def forward(self, x):
        output = self.conv(x)
        return self.DenseBlock(output)


# 解码器
class CNN_Decoder(nn.Module):
    def __init__(self, output_nc=1, kernel_size=3, stride=1):
        super(CNN_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            ConvLayer(64, 64, kernel_size, stride),
            CAMB(64),
            ConvLayer(64, 32, kernel_size, stride),
            CAMB(32),
            ConvLayer(32, 16, kernel_size, stride),
            CAMB(16),
            ConvLayer(16, output_nc, kernel_size, stride, is_last=True)
        )

    def forward(self, encoder_output):
        return self.decoder(encoder_output)


# 训练模型
class Train_Module(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1):
        super(Train_Module, self).__init__()
        self.encoder = Dense_Encoder(input_nc=input_nc, kernel_size=kernel_size, stride=stride)
        self.decoder = CNN_Decoder(output_nc=output_nc, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        encoder_feature = self.encoder(x)
        out = self.decoder(encoder_feature)
        return out


# 权重初始化
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# CBAM模块
class CAMB(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CAMB, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


# 绘制计算图的测试函数
def plot_graph(input=3, output=3):
    Encoder = Dense_Encoder(input_nc=input)
    Decoder = CNN_Decoder(output_nc=output)
    train_net = Train_Module(input_nc=input, output_nc=output)
    image = torch.randn((7, input, 256, 256))
    encode_feature = Encoder(image).detach()

    writer_e = SummaryWriter('logs_attention/net3/encoder')
    writer_e.add_graph(Encoder, image)
    writer_e.close()

    writer_d = SummaryWriter('logs_attention/net3/decoder')
    writer_d.add_graph(Decoder, encode_feature)
    writer_d.close()

    writer_t = SummaryWriter('logs_attention/net3/Training')
    writer_t.add_graph(train_net, image)
    writer_t.close()
    print("finish plot")
    # tensorboard --logdir=./net/


if __name__ == "__main__":
    plot_graph()
    # train_net = Train_Module()
    train_net = Train_Module(input_nc=3, output_nc=3)
    print("DenseFuse have {} paramerters in total".format(sum(x.numel() for x in train_net.parameters())))
    # RGB: DenseFuse have 74771 paramerters in total
    # GRAY: DenseFuse have 74193 paramerters in total
    # RGB: attrention DenseFuse have 78559 paramerters in total
