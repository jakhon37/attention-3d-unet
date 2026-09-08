import torch
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_chan, out_chan, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm3d(out_chan),
        nn.ReLU(inplace=True)
    )


def conv_stage(in_chan, out_chan):
    return nn.Sequential(
        conv_block(in_chan, out_chan),
        conv_block(out_chan, out_chan),
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = conv_stage(1, 2)
        self.enc2 = conv_stage(2, 2)
        self.enc2 = conv_stage(2, 2)
        # self.enc3 = conv_stage(2, 2)


        self.pool = nn.MaxPool3d(2, 2)

        self.dec2 = conv_stage(2, 2)
        self.dec1 = conv_stage(2, 2)
        self.conv_out = nn.Conv3d(2, 1, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.interpolate(enc1, (240,240,155), mode='trilinear', align_corners=False))
        # enc3 = self.enc3(F.interpolate(enc2, (320,320,320), mode='trilinear', align_corners=False))

        # dec2 = self.dec2(self.pool(enc3))
        dec1 = self.dec1(self.pool(enc2))
        out = self.conv_out(dec1)
        out = torch.sigmoid(out)
        return out


if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable

    torch.cuda.set_device(0)
    net = UNet().cuda().eval()
    # net = UNet().eval()

    # data = Variable(torch.randn(1, 64, 64, 155))
    data = Variable(torch.randn(1, 1, 64, 64, 155)).cuda()

    start_time = time.time()
    print("go")
    for _ in range(5):
        _ = net(data)
    print((time.time() - start_time)/5)
    summary(net, (1, 64, 64, 64))

    print(f'torch version: {torch.__version__}')

