import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BANDS = 4
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Sequential(
#         nn.ReplicationPad2d(1),  #镜像填充的值为1，并且stride=1,这样才能保证卷积后的特征跟原来保持一样。
#         nn.Conv2d(in_channels, out_channels, 3, stride=stride)
#     )

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

#第一个编码器称为LTHS编码器，用于陆地卫星特征提取.
class FEncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )

#第二个编码器称为残差编码器，用于学习参考日期和预测日期之间的特征差异.
class REncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS * 3, 32, 64, 128]
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3])
        )

#通过添加这两个编码器的特征映射，可以生成预测特征，最后重建解码器将这些高级特征恢复到原始的像素空间衍生预测。
class Decoder(nn.Sequential):
    def __init__(self):
        channels = [64, 32, 4, NUM_BANDS]
        super(Decoder, self).__init__(
            # conv3x3(channels[0], channels[1]),
            # nn.ReLU(True),
            # conv3x3(channels[1], channels[2]),
            # nn.ReLU(True),
            # nn.Conv2d(channels[2], channels[3], 1)
            conv3x3(channels[0], channels[1]),
            conv1x1(channels[1], channels[2]),
        )