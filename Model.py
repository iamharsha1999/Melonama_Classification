from torch import nn
import pretrainedmodels
import torch.nn.functional as F


class Resnet34(nn.Module):
    def __init__(self, model_name = 'resnext50_32x4d'):
        super().__init__()

        self.base_model = pretrainedmodels.__dict__['resnet34'](pretrained  = None)
    
        num_ftrs = self.base_model.last_linear.in_features
        self.final = nn.Sequential(
                                nn.Linear(num_ftrs, 32),
                                nn.ReLU(inplace = True),
                                nn.Linear(32,16),
                                nn.ReLU(inplace = True),
                                nn.Linear(16,8),
                                nn.ReLU(inplace = True),
                                nn.Linear(8,1)
            )
        self.base_model.avgpool = None
        self.base_model.last_linear = None

    def forward(self, x):
        bs,_,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)
        x = self.final(x)

        return x
