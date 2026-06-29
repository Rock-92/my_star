import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

class cSE(nn.Module):

    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


class sSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y


class scSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.cSE = cSE(in_channel)
        self.sSE = sSE(in_channel)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return torch.max(U_cse, U_sse)  # Taking the element-wise maximum


class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape

        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)

        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        x = x_maxpool + x_avgpool
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = inputs * x

        return outputs


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        x = self.conv(x)
        x = self.sigmoid(x)
        outputs = inputs * x

        return outputs


class CHannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CHannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SPatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPatialAttention, self).__init__()
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


class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = CHannelAttention(out_channels)
        self.sa = SPatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w
        weighted_data = feature * weighted
        conv_data = weighted_data.view(b, c, self.kernel_size, self.kernel_size, h, w)
        conv_data = conv_data.permute(0, 1, 4, 2, 5, 3).contiguous()
        conv_data = conv_data.view(b, c, h * self.kernel_size, w * self.kernel_size)
        return self.conv(conv_data)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ARFAM(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 2):
        super(ARFAM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes
        self.branch0 = nn.Sequential(
                RFAConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=1, dilation=1),
                scSE(2*inter_planes)
                )
        self.branch1 = nn.Sequential(
                RFAConv(in_planes, 2*inter_planes, kernel_size=3, stride=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1),
                scSE(2 * inter_planes)
        )
        self.branch2 = nn.Sequential(
                RFAConv(in_planes, 2*inter_planes, kernel_size=5, stride=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5, dilation=5),
                scSE(2 * inter_planes)
        )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()
        if stride != 1 or out_planes != in_planes:
            self.rescut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_planes))
        else:
            self.rescut = None
        self.rescbam = Res_CBAM_block(out_planes, out_planes)

    def forward(self,x):
        residual = x
        if self.rescut is not None:
            residual = self.rescut(x)

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)

        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = out + residual
        out = self.relu(out)
        out = self.rescbam(out)

        return out


class CLH(nn.Module):
    def __init__(self, low_channels):
        super(CLH, self).__init__()
        self.l_c = low_channels
        # low:
        self.low2 = nn.Sequential(
            nn.Conv2d(self.l_c, self.l_c, 1, bias=False),
            # nn.BatchNorm2d(self.l_c),
            # nn.ReLU(inplace=True),
            SpatialAttention1(kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x_low):
        low2_sigmoid = self.low2(x_low)
        return low2_sigmoid


class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention1, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x2 = torch.cat([x, pattn1], dim=1)  # B, C, 2, H, W
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CAFFM(nn.Module):
    def __init__(self, dim, reduction=2):
        super(CAFFM, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.x_changechannel = nn.Conv2d(in_channels=16, out_channels=dim,kernel_size=1)
        self.y_changechannel = nn.Conv2d(in_channels=128, out_channels=dim,kernel_size=1)
        self.resultchannel = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.CLH = CLH(dim)

    def forward(self, x, y):
        y = F.interpolate(y, size=(256,256), mode='bilinear', align_corners=True)
        x = self.x_changechannel(x)
        y = self.y_changechannel(y)
        initial = x + y
        caan = self.ca(initial)
        saan = self.sa(initial)
        pattn1 = saan + caan
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * y + (1 - pattn2) * x
        result = self.conv(result)
        result = self.resultchannel(result)
        return result


class DNRFANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,
                 netdepth=3, scale_method='deconv', deep_supervision=False, name='DNRFANet'):

        super(DNRFANet, self).__init__()
        self.netdepth = netdepth
        self.scale_method = scale_method.lower()
        self.deep_supervision = True if deep_supervision.lower() == 'true' else False
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.caffm = CAFFM(input_channels)
        self.caffmchannel = nn.Conv2d(64, 1, 1)

        self._gen_scale_conv(netdepth, scale_method, nb_filter)

        self._gen_nested_conv(netdepth, block, input_channels, nb_filter, num_blocks)

        self._gen_parallel_1x1conv(netdepth, nb_filter)

        self.__setattr__("conv0_{}_final".format(netdepth - 1),
                         self._make_layer(block, nb_filter[0] * netdepth, nb_filter[0]))

        if self.deep_supervision:
            self._gen_multi_finalconv(netdepth, nb_filter, num_classes)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def _gen_scale_conv(self, netdepth, scale_method, nb_filter):
        if scale_method.lower() == 'biinterp':
            # up: bilinear-interp
            self.__setattr__("up", nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.__setattr__("up_4", nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
            self.__setattr__("up_8", nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
            self.__setattr__("up_16", nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))

            # down: bilinear-interp
            self.__setattr__("down", nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True))

        elif scale_method.lower() == 'deconv':
            # up: deconv
            for i in range(1, netdepth):
                conv_module_up = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=int(pow(2, i + 1)),
                                                    stride=int(pow(2, i)), padding=int(pow(2, i - 1)))
                self.__setattr__("up" if i == 1 else "up_{}".format(int(pow(2, i))), conv_module_up)
            for i in range(1, netdepth):
                for j in range(netdepth - i):
                    conv_module_up = nn.ConvTranspose2d(nb_filter[i], nb_filter[i], kernel_size=4, stride=2, padding=1)
                    self.__setattr__("up_{}_{}".format(i, j), conv_module_up)
            # down: conv
            for i in range(netdepth - 1):
                for j in range(netdepth - 1 - i):
                    conv_module_down = nn.Conv2d(nb_filter[i], nb_filter[i], kernel_size=4, stride=2, padding=1)
                    self.__setattr__("down_{}_{}".format(i, j), conv_module_down)

        elif scale_method.lower() == 'nearest':
            # up: nearest-interp
            self.__setattr__("up", nn.Upsample(scale_factor=2, mode='nearest', align_corners=None))
            self.__setattr__("up_4", nn.Upsample(scale_factor=4, mode='nearest', align_corners=None))
            self.__setattr__("up_8", nn.Upsample(scale_factor=8, mode='nearest', align_corners=None))
            self.__setattr__("up_16", nn.Upsample(scale_factor=16, mode='nearest', align_corners=None))

            # down: bilinear-interp
            self.__setattr__("down", nn.Upsample(scale_factor=0.5, mode='nearest', align_corners=None))
        else:
            raise Exception('wrong scale method')

    def _gen_nested_conv(self, netdepth, block, input_channels, nb_filter, num_blocks):
        for i in range(netdepth):
            for j in range(netdepth - i):
                if i == 0 and j == 0:
                    conv_module = self._make_layer(block, input_channels, nb_filter[0])
                elif j == 0:
                    conv_module = self._make_layer(block, nb_filter[i - 1], nb_filter[i], num_blocks[i - 1])
                elif i == 0:
                    conv_module = self._make_layer(block, nb_filter[i] * j + nb_filter[i + 1], nb_filter[i])
                else:
                    conv_module = self._make_layer(block, nb_filter[i] * j + nb_filter[i + 1] + nb_filter[i - 1],
                                                   nb_filter[i], num_blocks[i - 1])
                self.__setattr__("conv{}_{}".format(i, j), conv_module)

    def _gen_parallel_1x1conv(self, netdepth, nb_filter):
        for i in range(1, netdepth):
            conv_module_1x1 = nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1)
            self.__setattr__("conv0_{}_1x1".format(i), conv_module_1x1)

    def _gen_multi_finalconv(self, netdepth, nb_filter, num_classes):
        for i in range(1, netdepth):
            conv_module_final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.__setattr__("final{}".format(i), conv_module_final)

    def forward(self, input):
        netdepth = self.netdepth
        scale_method = self.scale_method
        assert netdepth >= 3 and netdepth <= 5, "network depth must >=3 and <=5"

        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.down_0_0(x0_0)) if scale_method == 'deconv' else self.conv1_0(self.down(x0_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_1_0(x1_0)], 1)) if scale_method == 'deconv' \
            else self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.down_1_0(x1_0)) if scale_method == 'deconv' else self.conv2_0(self.down(x1_0))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_2_0(x2_0), self.down_0_1(x0_1)], 1)) if scale_method == 'deconv' \
            else self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_1_1(x1_1)], 1)) if scale_method == 'deconv' \
            else self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        if netdepth >= 4:
            x3_0 = self.conv3_0(self.down_2_0(x2_0)) if scale_method == 'deconv' else self.conv3_0(self.down(x2_0))

            x2_1 = self.conv2_1(
                torch.cat([x2_0, self.up_3_0(x3_0), self.down_1_1(x1_1)], 1)) if scale_method == 'deconv' \
                else self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))

            x1_2 = self.conv1_2(
                torch.cat([x1_0, x1_1, self.up_2_1(x2_1), self.down_0_2(x0_2)], 1)) if scale_method == 'deconv' \
                else self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))

            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up_1_2(x1_2)], 1)) if scale_method == 'deconv' \
                else self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))


        if netdepth >= 5:
            x4_0 = self.conv4_0(self.down_3_0(x3_0)) if scale_method == 'deconv' else self.conv4_0(self.down(x3_0))
            x3_1 = self.conv3_1(
                torch.cat([x3_0, self.up_4_0(x4_0), self.down_2_1(x2_1)], 1)) if scale_method == 'deconv' \
                else self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
            x2_2 = self.conv2_2(
                torch.cat([x2_0, x2_1, self.up_3_1(x3_1), self.down_1_2(x1_2)], 1)) if scale_method == 'deconv' \
                else self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
            x1_3 = self.conv1_3(
                torch.cat([x1_0, x1_1, x1_2, self.up_2_2(x2_2), self.down_0_3(x0_3)], 1)) if scale_method == 'deconv' \
                else self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up_1_3(x1_3)], 1)) if scale_method == 'deconv' \
                else self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if netdepth == 3:
            Final_x0_2 = self.conv0_2_final(torch.cat([self.up_4(self.conv0_2_1x1(x2_0)),
                                                       self.up(self.conv0_1_1x1(x1_1)), x0_2], 1))

            if self.deep_supervision:
                caffm1 = self.caffm(x1_1, x2_0)
                caffm2 = self.caffm(x0_2, caffm1)
                output = self.caffmchannel(caffm2)
                output1 = self.final1(x0_1)
                output2 = self.final2(Final_x0_2)
                return [output, output1, output2]
            else:
                output = self.final(Final_x0_2)
                return output

        elif netdepth == 4:
            Final_x0_3 = self.conv0_3_final(torch.cat([self.up_8(self.conv0_3_1x1(x3_0)),
                                                       self.up_4(self.conv0_2_1x1(x2_1)),
                                                       self.up(self.conv0_1_1x1(x1_2)), x0_3], 1))

            if self.deep_supervision:
                caffm1 = self.caffm(x2_1, x3_0)
                caffm2 = self.caffm(x1_2, caffm1)
                caffm3 = self.caffm(x0_3, caffm2)
                output = self.caffmchannel(caffm3)
                output1 = self.final1(x0_1)
                output2 = self.final2(x0_2)
                output3 = self.final3(Final_x0_3)
                return [output, output1, output2, output3]
            else:
                output = self.final(Final_x0_3)
                return output

        elif netdepth == 5:
            Final_x0_4 = self.conv0_4_final(
                torch.cat([self.up_16(self.conv0_4_1x1(x4_0)), self.up_8(self.conv0_3_1x1(x3_1)),
                           self.up_4(self.conv0_2_1x1(x2_2)), self.up(self.conv0_1_1x1(x1_3)), x0_4], 1))

            if self.deep_supervision:
                caffm1 = self.caffm(x3_1, x4_0)
                caffm2 = self.caffm(x2_2, caffm1)
                caffm3 = self.caffm(x1_3, caffm2)
                caffm4 = self.caffm(x0_4, caffm3)
                output = self.caffmchannel(caffm4)
                output1 = self.final1(x0_1)
                output2 = self.final2(x0_2)
                output3 = self.final3(x0_3)
                output4 = self.final4(Final_x0_4)
                return [output, output1, output2, output3, output4]
            else:
                output = self.final(Final_x0_4)
                return output

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()

        # check loaded parameters and created model parameters
        msg = 'If you see this, your model does not fully load the ' + \
              'pre-trained weight. Please make sure ' + \
              'you have correctly specified --arch xxx ' + \
              'or set the correct --num_classes for your own dataset.'
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k) + msg)
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
