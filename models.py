import torch
from torch import nn
import torchvision.models


class ModifiedVGG(nn.Module):

    def __init__(self):
        super(ModifiedVGG, self).__init__()

        trained_vgg = torchvision.models.vgg19(pretrained=True)
        m = list(trained_vgg.features.children())
        # Conv. 1
        self.conv1_1 = m[0]
        self.relu1_1 = m[1]
        self.conv1_2 = m[2]
        self.relu1_2 = m[3]
        self.a_pool1 = self.pool(m[4])
        # Conv. 2
        self.conv2_1 = m[5]
        self.relu2_1 = m[6]
        self.conv2_2 = m[7]
        self.relu2_2 = m[8]
        self.a_pool2 = self.pool(m[9])

        # Conv. 3
        self.conv3_1 = m[10]
        self.relu3_1 = m[11]
        self.conv3_2 = m[12]
        self.relu3_2 = m[13]
        self.conv3_3 = m[14]
        self.relu3_3 = m[15]
        self.conv3_4 = m[16]
        self.relu3_4 = m[17]
        self.a_pool3 = self.pool(m[18])

        # Conv. 4
        self.conv4_1 = m[19]
        self.relu4_1 = m[20]
        self.conv4_2 = m[21]
        self.relu4_2 = m[22]
        self.conv4_3 = m[23]
        self.relu4_3 = m[24]
        self.conv4_4 = m[25]
        self.relu4_4 = m[26]
        self.a_pool4 = self.pool(m[27])

        # Conv. 5
        self.conv5_1 = m[28]
        self.relu5_1 = m[29]
        self.conv5_2 = m[30]
        self.relu5_2 = m[31]
        self.conv5_3 = m[32]
        self.relu5_3 = m[33]
        self.conv5_4 = m[34]
        self.relu5_4 = m[35]
        self.a_pool5 = self.pool(m[36])

    def forward(self, x):
        # Conv1
        conv1_1 = self.conv1_1(x)
        x = self.relu1_1(conv1_1)
        conv1_2 = self.conv1_2(x)
        x = self.relu1_2(conv1_2)
        x = self.a_pool1(x)
        # Conv2
        conv2_1 = self.conv2_1(x)
        x = self.relu2_1(conv2_1)
        conv2_2 = self.conv2_2(x)
        x = self.relu2_2(conv2_2)
        x = self.a_pool2(x)
        # Conv3
        conv3_1 = self.conv3_1(x)
        x = self.relu3_1(conv3_1)
        conv3_2 = self.conv3_2(x)
        x = self.relu3_2(conv3_2)
        conv3_3 = self.conv3_3(x)
        x = self.relu3_3(conv3_3)
        conv3_4 = self.conv3_4(x)
        x = self.relu3_4(conv3_4)
        x = self.a_pool3(x)

        # Conv4
        conv4_1 = self.conv4_1(x)
        x = self.relu4_1(conv4_1)
        conv4_2 = self.conv4_2(x)
        x = self.relu4_2(conv4_2)
        conv4_3 = self.conv4_3(x)
        x = self.relu4_3(conv4_3)
        conv4_4 = self.conv4_4(x)
        x = self.relu4_4(conv4_4)
        x = self.a_pool4(x)

        # Conv5
        conv5_1 = self.conv5_1(x)
        x = self.relu5_1(conv5_1)
        conv5_2 = self.conv5_2(x)
        x = self.relu5_2(conv5_2)
        conv5_3 = self.conv5_3(x)
        x = self.relu5_3(conv5_3)
        conv5_4 = self.conv5_4(x)
        x = self.relu5_4(conv5_4)
        x = self.a_pool5(x)

        # Append all output layers
        returns = {
            'conv1_1': conv1_1, 'conv1_2': conv1_2,
            'conv2_1': conv2_1, 'conv2_2': conv2_2,
            'conv3_1': conv3_1, 'conv3_2': conv3_2,
            'conv3_3': conv3_3, 'conv3_4': conv3_4,
            'conv4_1': conv4_1, 'conv4_2': conv4_2,
            'conv4_3': conv4_3, 'conv4_4': conv4_4,
            'conv5_1': conv5_1, 'conv5_2': conv5_2,
            'conv5_3': conv5_3, 'conv5_4': conv5_4,
        }
        return x, returns

    def pool(self, max_pool):
        return nn.AvgPool2d(kernel_size=max_pool.kernel_size,
                            stride=max_pool.stride,
                            padding=max_pool.padding)
