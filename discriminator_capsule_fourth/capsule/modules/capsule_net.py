import torch
import torch.nn as nn
import torch.nn.functional as F
from .capsule_layers import PrimaryCapsule, MECapsule
from .activations import squash
from torchvision import models


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CapsuleAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CapsuleAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ResNetLayers(nn.Module):
    def __init__(self, is_freeze=False):
        super(ResNetLayers, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # print('self.model(origin):', self.model)
        delattr(self.model, 'layer4')  # delattr(x,'y') ---> del x.y
        delattr(self.model, 'avgpool')
        delattr(self.model, 'fc')
        # print('self.model(processed):', self.model)

        if is_freeze:
            for index, p in enumerate(self.model.parameters()):
                if index == 15:
                    break
                p.requires_grad = False

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        return output


class VGGLayers(nn.Module):
    def __init__(self, is_freeze=True):
        super(VGGLayers, self).__init__()
        self.model = models.vgg11(pretrained=True).features[:11]

        if is_freeze:
            for i in range(4):
                for p in self.model[i].parameters():
                    p.requires_grad = False

    def forward(self, x):
        # x = [B, 3, 224, 224]
        return self.model(x)  # [B, 256, 20, 20]


backbone = {'vgg': VGGLayers, 'resnet': ResNetLayers}


class MECapsuleNet(nn.Module):
    """
	A Capsule Network on Micro-expression.
	:param input_size: data size = [channels, width, height]
	:param classes: number of classes
	:param routings: number of routing iterations
	Shape:
		- Input: (batch, channels, width, height), optional (batch, classes) .
		- Output:((batch, classes), (batch, channels, width, height))
	"""

    def __init__(self, input_size, classes, routings, conv_name='resnet', is_freeze=True):
        super(MECapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        self.conv = backbone[conv_name](is_freeze)  # Figure CM1
        # self.ca = ChannelAttention(256)
        # self.sa = SpatialAttention()
        # self.conv2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=9, stride=1, padding=0)
        self.eca = eca_layer(256)
        # self.ca = ChannelAttention(256)
        # self.sa = SpatialAttention()

        self.primarycaps = PrimaryCapsule(256, 32 * 8, 8, kernel_size=9, stride=2, padding=0)
        # self.ca = CapsuleAttention(1152)
        # self.sa = CapsuleAttention()

        self.digitcaps = MECapsule(in_num_caps=32 * 6 * 6,
                                   in_dim_caps=8,
                                   out_num_caps=self.classes,
                                   out_dim_caps=16,
                                   routings=routings)  # [32,1152,8] -> [32,3,16]
        self.relu = nn.ReLU()
        # self.discriminator = nn.Sequential(nn.Linear(32 * 36 * 8, 16 * 18 * 8),
        #                                    nn.ReLU(True), nn.Dropout(),
        #                                    nn.Linear(16 * 18 * 8, 18 * 4),
        #                                    nn.ReLU(True), nn.Dropout(),
        #                                    nn.Linear(18 * 4, 2))
        self.discriminator = nn.Sequential(nn.Linear(32 * 36 * 8, 16 * 36),
                                           nn.ReLU(True), nn.Dropout(),
                                           nn.Linear(16 * 36, 9 * 4),
                                           nn.ReLU(True), nn.Dropout(),
                                           nn.Linear(9 * 4, 2))

    def forward(self, x, y=None, y_domain=None):
        x = self.conv(x)
        # x = self.relu(self.conv2(x))
        # print(x.size())
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.relu(self.conv1(x))
        x = self.eca(x)
        # print(x.size())
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.primarycaps(x)  # [32, 1152, 8]
        # print('hello', x.size())

        # x = self.ca(x) * x
        # x = self.sa(x) * x
        out_c = self.digitcaps(x)  # [32, 3, 16 ]
        out_d = self.discriminator(x.view(x.size(0), -1))  # [32,1192]
        # print(out_d.size())
        length = out_c.norm(dim=-1)
        return length, out_d
