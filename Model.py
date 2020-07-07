import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F


class Resnet34(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = pretrainedmodels.__dict__['resnet34'](pretrained  = None)

        num_ftrs = self.base_model.last_linear.in_features
        self.final = nn.Sequential(nn.Linear(num_ftrs, 1),
                                                                    nn.Sigmoid())
        self.base_model.avgpool = None
        self.base_model.last_linear = None

    def forward(self, x):
        bs,_,_,_ = x.shape
        x = self.base_model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)
        x = self.final(x)

        return x

class SemiUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ## Conv Layers for Feature Extraction
        self.c1 = nn.Conv2d(in_channels, 64, 3)
        self.b1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 64, 3)
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(64, 128, 5)
        self.b3 = nn.BatchNorm2d(128)
        self.c4 = nn.Conv2d(128,128, 5)
        self.b4 = nn.BatchNorm2d(128)

        self.d1 = nn.MaxPool2d(2)
        self.c5 = nn.Conv2d(128,256, 5)

        self.d2 = nn.MaxPool2d(2)
        self.c6 = nn.Conv2d(256, 512, 5)

        ## FC Layers for Classification
        self.final = nn.Sequential(
                                nn.Linear(512, 32),
                                nn.ReLU(inplace = True),
                                nn.Linear(32,16),
                                nn.ReLU(inplace = True),
                                nn.Linear(16,8),
                                nn.ReLU(inplace = True),
                                nn.Linear(8,1),
                                nn.Sigmoid()
            )


    def forward(self,x):

        bs,_,_,_ = x.shape

        x  = self.c1(x)
        x  = self.b1(x)
        x  = F.relu(x)

        x  = self.c2(x)
        x  = self.b2(x)
        x  = F.relu(x)

        x  = self.c3(x)
        x  = self.b4(x)
        x  = F.relu(x)

        x  = self.c4(x)
        x  = self.b4(x)
        x  = F.relu(x)

        x  = self.d1(x)
        x  = self.c5(x)
        x  = F.relu(x)

        x  = self.d2(x)
        x  = self.c6(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)

        x = self.final(x)

        return x

class SEResnext50_32x4d(nn.Module):

    def __init__(self):

        super().__init__()

        self.base_model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained = 'imagenet')

        num_ftrs = self.base_model.last_linear.in_features
        self.final = nn.Sequential(
                            nn.Linear(num_ftrs, 1),
                            nn.Sigmoid()
        )


    def forward(self, x):

        bs,_,_,_ = x.shape
        x = self.base_model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)
        x = self.final(x)

        return x 





            

