import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import discriminator
from ssim import msssim
NUM_BANDS = 4


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),  #镜像填充的值为1，并且stride=1,这样才能保证卷积后的特征跟原来保持一样。
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


#上采样，bilinear双线性插值，倍数是scale_factor倍
def interpolate(inputs, size=None, scale_factor=None):
    return F.interpolate(inputs, size=size, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True)

#特征损失，与论文中不太一样。
class Pretrained(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128, 128]
        super(Pretrained, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True),
            conv3x3(channels[3], channels[4]),
            nn.ReLU(True)
        )

class CompoundLoss(nn.Module):
    def __init__(self, device,pretrained, alpha=0.8, normalize=True):
        super(CompoundLoss, self).__init__()
        self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize
        # l1损失
        self.loss = nn.L1Loss()

        self.advloss = AdversarialLoss(device)

    def forward(self, prediction, target):

        # 内容损失、特征损失、视觉损失。F.mse_loss:均方损失函数。
        return (F.mse_loss(prediction, target) +
                F.mse_loss(self.pretrained(prediction), self.pretrained(target)) +
                self.alpha * (1.0 - msssim(prediction, target,
                                           normalize=self.normalize))
                )


class AdversarialLoss(nn.Module):
    def __init__(self, device, num_gpu=1, gan_k=1,
                 lr_dis=1e-4):

        super(AdversarialLoss, self).__init__()
        self.gan_k = gan_k
        self.device = device
        self.discriminator = discriminator.Discriminator().to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        self.optimizer = optim.Adam(
            self.discriminator.parameters(),
            betas=(0, 0.9), eps=1e-8, lr=lr_dis
        )

        self.bce_loss = torch.nn.BCELoss().to(self.device)

    def forward(self, fake, real):
        fake_detach = []
        d_fake_for_g = 0
        for i in range(len(fake)):
            fake_detach.append(fake[i].detach())
        for i in range(len(fake_detach)):
            for _ in range(self.gan_k):
                self.optimizer.zero_grad()
                d_fake = self.discriminator(fake_detach[i])
                d_real = self.discriminator(real)

                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.
            # Discriminator update
            loss_d.backward()
            self.optimizer.step()
        for step in range(len(fake)):
            d_fake_for_g += self.discriminator(fake[step])
        loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)
        # Generator loss
        return loss_g

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict