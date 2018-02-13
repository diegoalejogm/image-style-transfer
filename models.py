import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models


class ModifiedVGG(nn.Module):

    def __init__(self):
        super(ModifiedVGG, self).__init__()

        trained_vgg = torchvision.models.vgg19(pretrained=True)
        m = list(trained_vgg.features.children())
        # Conv. 1
        self.conv1_1 = m[0]
        self.conv1_2 = m[2]
        self.pool1 = self.pool_layer(m[4])
        
        # Conv. 2
        self.conv2_1 = m[5]
        self.conv2_2 = m[7]
        self.pool2 = self.pool_layer(m[9])

        # Conv. 3
        self.conv3_1 = m[10]
        self.conv3_2 = m[12]
        self.conv3_3 = m[14]
        self.conv3_4 = m[16]
        self.pool3 = self.pool_layer(m[18])

        # Conv. 4
        self.conv4_1 = m[19]
        self.conv4_2 = m[21]
        self.conv4_3 = m[23]
        self.conv4_4 = m[25]
        self.pool4 = self.pool_layer(m[27])

        # Conv. 5
        self.conv5_1 = m[28]
        self.conv5_2 = m[30]
        self.conv5_3 = m[32]
        self.conv5_4 = m[34]
        self.pool5 = self.pool_layer(m[36])
        
        if torch.cuda.is_available():
            self.cuda()
            
        self.disable_grad()
        
#         self.relu1_1 = m[1]
#         self.relu1_2 = m[3]
#         self.relu2_1 = m[6]
#         self.relu2_2 = m[8]
#         self.relu3_1 = m[11]
#         self.relu3_2 = m[13]
#         self.relu3_3 = m[15]
#         self.relu3_4 = m[17]
#         self.relu4_1 = m[20]
#         self.relu4_2 = m[22]
#         self.relu4_3 = m[24]
#         self.relu4_4 = m[26]
#         self.relu5_1 = m[29]
#         self.relu5_2 = m[31]
#         self.relu5_3 = m[33]
#         self.relu5_4 = m[35]

    def forward(self, x):
        out = {}
        
        out['relu1_1'] = F.relu(self.conv1_1(x))
        out['relu1_2'] = F.relu(self.conv1_2(out['relu1_1']))
        out['p1'] = self.pool1(out['relu1_2'])
        out['relu2_1'] = F.relu(self.conv2_1(out['p1']))
        out['relu2_2'] = F.relu(self.conv2_2(out['relu2_1']))
        out['p2'] = self.pool2(out['relu2_2'])
        out['relu3_1'] = F.relu(self.conv3_1(out['p2']))
        out['relu3_2'] = F.relu(self.conv3_2(out['relu3_1']))
        out['relu3_3'] = F.relu(self.conv3_3(out['relu3_2']))
        out['relu3_4'] = F.relu(self.conv3_4(out['relu3_3']))
        out['p3'] = self.pool3(out['relu3_4'])
        out['relu4_1'] = F.relu(self.conv4_1(out['p3']))
        out['relu4_2'] = F.relu(self.conv4_2(out['relu4_1']))
        out['relu4_3'] = F.relu(self.conv4_3(out['relu4_2']))
        out['relu4_4'] = F.relu(self.conv4_4(out['relu4_3']))
        out['p4'] = self.pool4(out['relu4_4'])
        out['relu5_1'] = F.relu(self.conv5_1(out['p4']))
        out['relu5_2'] = F.relu(self.conv5_2(out['relu5_1']))
        out['relu5_3'] = F.relu(self.conv5_3(out['relu5_2']))
        out['relu5_4'] = F.relu(self.conv5_4(out['relu5_3']))
        out['p5'] = self.pool5(out['relu5_4'])
    
        return out
    
    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def pool_layer(self, max_pool):
        return nn.AvgPool2d(kernel_size=max_pool.kernel_size,
                            stride=max_pool.stride,
                            padding=max_pool.padding)
