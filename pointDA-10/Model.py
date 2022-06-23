from model_utils import *
import pdb
import os
import torch.nn.functional as F

# Channel Attention


# Classifier
class Pointnet_c(nn.Module):
    def __init__(self, num_class=10):
        super(Pointnet_c, self).__init__()
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.fc(x)
        return x


class FE(nn.Module):
    def __init__(self):
        super(FE, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv1_bn= nn.BatchNorm2d(64)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv2_bn= nn.BatchNorm2d(64)
        # SA Node Module
        self.conv3 = adapt_layer_off()  # (64->128)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv4_bn= nn.BatchNorm2d(128)
        self.conv5 = conv_2d(128, 1024, 1)

        self.conv5_bn= nn.BatchNorm2d(1024)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node=False): #x=bs*3*1024*1
        x_loc = x.squeeze(-1)#bs*3*1024

        transform = self.trans_net1(x)#bs*3*3
        x = x.transpose(2, 1) #bs*1024*3*1

        x = x.squeeze(-1)#bs*1024*3
        x = torch.bmm(x, transform)#bs*1024*3
        x = x.unsqueeze(3)#bs*1024*3*1
        x = x.transpose(2, 1)#bs*3*1024*1
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv2(x)#bs*64*1024*1
        x = self.conv2_bn(x)
        transform = self.trans_net2(x)#bs*64*64
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)#bs*64*1024*1

        x, node_fea, node_off = self.conv3(
            x, x_loc
        )  #Output=bs*128*1024*1; x = [B, dim, num_node, 1]/[64, 64, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x, _ = torch.max(x, dim=2, keepdim=False)

        x = x.squeeze(-1)#bs*1024

        x = self.bn1(x)#bs*1024

        return x#, node_fea, node_off
class update_M(torch.nn.Module):
    def __init__(self,subspace_dim,source_subspace,target_subspace,init_M):
        super(update_M, self).__init__()
        self.subspace_dim=subspace_dim
        
        self.m=nn.Linear(init_M.shape[0],init_M.shape[1],bias =False)#nn.Linear(self.subspace_dim,self.subspace_dim,bias=False)
        self.m.weight.data= init_M.clone()
        self.xs = source_subspace
        self.xt = target_subspace
    def eye_like(self,tensor):
        return torch.eye(*tensor.size(), out=torch.empty_like(tensor))
    def forward(self,x=None):
        M= self.m.weight
        xa_updated= torch.mm(self.xt,M)
        W= torch.mm(xa_updated,torch.transpose(self.xs,1,0))
        return M,W
   