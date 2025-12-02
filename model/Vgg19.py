import torch
from torch import nn
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, rgb_range=1):
        super(Vgg19, self).__init__()
        
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()

        ### 改变通道
        self.slice1.add_module(str(0), nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        for x in range(1,30):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
        

    def forward(self, X):
        h =X
        h_relu5_1 = self.slice1(h)
        return h_relu5_1


if __name__ == '__main__':
    vgg19 = Vgg19(requires_grad=False)
    print('111')