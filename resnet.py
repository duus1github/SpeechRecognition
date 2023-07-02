#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:resnet.py
@time:2022/05/23
@desc:''

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        out = self.bn2(out)

        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        self.blk1 = ResBlk(16, 32, stride=3)  # [b,32,h,w]
        self.blk2 = ResBlk(32, 64, stride=3)  # [b,64,h,w]
        self.blk3 = ResBlk(64, 128, stride=2)  # [b,128,h,w]
        self.blk4 = ResBlk(128, 256, stride=2)  # [b,256,h,w]
        self.outlayer = nn.Linear(256 * 3 * 3, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():
    blk = ResBlk(64, 128)  # 输入64，输出128
    tmp = torch.randn(2, 64, 224, 224)

    out = blk(tmp)
    print('out1:',out)
    model = ResNet18(5)
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    print('out2:',out)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('p:',p)

if __name__ == '__main__':
    main()
