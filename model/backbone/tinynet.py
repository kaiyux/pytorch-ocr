import torch
import torch.nn as nn


class TinyNet(nn.Module):

    def __init__(self):
        super(TinyNet, self).__init__()

        self.pool = nn.MaxPool2d(
            2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1,
                               dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1,
                               dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 256, 3, stride=1, padding=1,
                               dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1,
                               dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1,
                               dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1,
                               dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, img):
        conv1 = self.conv1(img)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        conv2 = self.bn2(conv2)
        conv2 = self.relu(conv2)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        conv3 = self.bn3(conv3)
        conv3 = self.relu(conv3)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)
        conv4 = self.bn4(conv4)
        conv4 = self.relu(conv4)
        pool4 = self.pool(conv4)

        conv5 = self.conv5(pool4)
        conv5 = self.bn5(conv5)
        conv5 = self.relu(conv5)
        pool5 = self.pool(conv5)

        conv6 = self.conv6(pool5)
        conv6 = self.bn6(conv6)
        conv6 = self.relu(conv6)

        return conv6
