import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import h5py
import math
import torch.nn.functional as F


class DML(nn.Module):
    def __init__(self,output_size):
        super(DML, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), (1,1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d((3, 1), (3, 1), 0),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5),(1,1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d((2, 1), (2, 1), 0),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), (1,1), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )

        self.cnn6 = nn.Sequential(
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.tower1 =nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, output_size),
        )
    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = self.cnn5(out)
        h_shared = self.cnn6(out)
        out1 = self.tower1(h_shared)
        return out1

class chenbodata_muilt(nn.Module):
    def __init__(self,output_size1,output_size2):
        super(chenbodata_muilt, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), (1,1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d((3, 1), (3, 1), 0),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5),(1,1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d((2, 1), (2, 1), 0),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), (1,1), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )

        self.cnn6 = nn.Sequential(
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.tower1 =nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, output_size1),
        )
        self.tower2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, output_size2),
        )
    def forward(self, x):
        out = self.cnn1(x)
        # print('cnn1',out.shape)
        out = self.cnn2(out)
        # print('cnn2',out.shape)
        out = self.cnn3(out)
        # print('cnn3',out.shape)
        out = self.cnn4(out)
        out = self.cnn5(out)
        # print('cnn4', out.shape)
        h_shared = self.cnn6(out)
        # print('cnn5', h_shared.shape)
        out1 = self.tower1(h_shared)
        out2 = self.tower2(h_shared)
        # print('tower', out1.shape)
        # out1 = F.softmax(out1, dim=1)
        return out1,out2
class chenbodata(nn.Module):
    def __init__(self,output_size):
        super(chenbodata, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), (1,1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d((3, 1), (3, 1), 0),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5),(1,1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d((2, 1), (2, 1), 0),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), (1,1), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )

        self.cnn6 = nn.Sequential(
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.tower1 =nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, output_size),
        )
    def forward(self, x):
        out = self.cnn1(x)
        # print('cnn1',out.shape)
        out = self.cnn2(out)
        # print('cnn2',out.shape)
        out = self.cnn3(out)
        # print('cnn3',out.shape)
        out = self.cnn4(out)
        out = self.cnn5(out)
        # print('cnn4', out.shape)
        h_shared = self.cnn6(out)
        # print('cnn5', h_shared.shape)
        out1 = self.tower1(h_shared)
        # print('tower', out1.shape)
        # out1 = F.softmax(out1, dim=1)
        return out1

class Classifier_singal_out(nn.Module):
    def __init__(self,output_size):
        super(Classifier_singal_out, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), (1,1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d((3, 1), (3, 1), 0),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5),(1,1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d((2, 1), (2, 1), 0),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), (1,1), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )

        self.tower1 = nn.Sequential(
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, output_size),

        )

    def forward(self, x):
        out = self.cnn1(x)
        # print('cnn1',out.shape)
        out = self.cnn2(out)
        # print('cnn2',out.shape)
        out = self.cnn3(out)
        # print('cnn3',out.shape)
        out = self.cnn4(out)
        # print('cnn4', out.shape)
        h_shared = self.cnn5(out)
        # print('cnn5', h_shared.shape)
        out1 = self.tower1(h_shared)
        # print('tower', out1.shape)
        # out1 = F.softmax(out1, dim=1)
        return out1

class Classifier_singal_out_dropout(nn.Module):
    def __init__(self,output_size):
        super(Classifier_singal_out_dropout, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), (1,1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d((3, 1), (3, 1), 0),
        )
        self.droplayer1 = nn.Dropout(p=0.2)

        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5),(1,1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d((2, 1), (2, 1), 0),
        )

        self.droplayer2 = nn.Dropout(p=0.3)

        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), (1,1), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.droplayer3 = nn.Dropout(p=0.4)
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.droplayer4 = nn.Dropout(p=0.5)
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d((1, 1), (1, 1), 0),
        )
        self.droplayer5 = nn.Dropout(p=0.5)
        self.tower1 = nn.Sequential(
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        out = self.cnn1(x)
        out = self.droplayer1(out)

        out = self.cnn2(out)
        out = self.droplayer2(out)

        out = self.cnn3(out)
        out = self.droplayer3(out)

        out = self.cnn4(out)
        out = self.droplayer4(out)
        # print('cnn4', out.shape)
        h_shared = self.cnn5(out)
        h_shared = self.droplayer5(h_shared)
        # print('cnn5', h_shared.shape)
        out1 = self.tower1(h_shared)
        # print('tower', out1.shape)
        # out1 = F.softmax(out1, dim=1)
        return out1



class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         # torch.nn.MaxPool2d(kernel_size, stride, padding)
#
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(1, 32, (5,5), (1,1), (2, 2)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             # nn.MaxPool2d((3, 1), (3, 1), 0),
#         )
#         self.cnn2 = nn.Sequential(
#             nn.Conv2d(32, 64, (5, 5),(1,1), (2, 2)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             # nn.MaxPool2d((2, 1), (2, 1), 0),
#         )
#         self.cnn3 = nn.Sequential(
#             nn.Conv2d(64, 128, (5, 5), (1,1), (2, 2)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             # nn.MaxPool2d((1, 1), (1, 1), 0),
#         )
#         self.cnn4 = nn.Sequential(
#             nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             # nn.MaxPool2d((1, 1), (1, 1), 0),
#         )
#         self.cnn5 = nn.Sequential(
#             nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             # nn.MaxPool2d((1, 1), (1, 1), 0),
#         )
#
#         self.tower1 = nn.Sequential(
#             nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#
#             # nn.Linear(1024, 512),
#             # nn.ReLU(),
#             # nn.Dropout(),
#             # nn.Linear(512, 100),
#             # nn.ReLU(),
#             # nn.Dropout(),
#             nn.Linear(1024, output1_size),
#         )
#         self.tower2 = nn.Sequential(
#             nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#
#
#             nn.Linear(1024, output2_size)
#         )
#
#     def forward(self, x):
#         out = self.cnn1(x)
#         out = self.cnn2(out)
#         out = self.cnn3(out)
#         out = self.cnn4(out)
#         h_shared = self.cnn5(out)
#         out1 = self.tower1(h_shared)
#         out2 = self.tower2(h_shared)
#         return out1, out2
