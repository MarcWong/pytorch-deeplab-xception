import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 1280 #128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.last_conv = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                       nn.Conv2d(256, 256, 3, 1, 1),
                                       nn.Dropout(0.3),
                                       nn.Upsample(scale_factor=2),
                                       nn.Conv2d(256, 128, 3, 1, 1),
                                       nn.Conv2d(128, 128, 3, 1, 1),
                                       nn.Dropout(0.3),
                                       nn.Upsample(scale_factor=2),
                                       nn.Conv2d(128, 64, 3, 1, 1),
                                       nn.Dropout(0.3),
                                       nn.Upsample(scale_factor=2),
                                       nn.Conv2d(64, num_classes, kernel_size=1))
        self._init_weight()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        #x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)