import torch
import torch.nn as nn
import point_utils
import torch.nn.functional as F
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu'):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'tanh':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.Tanh()
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )


    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU()
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x



class transform_net(nn.Module):
    def __init__(self, in_ch, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = nn.Linear(256, K*K)
    


    def forward(self, x):
        #bs*3*1024*1
        x = self.conv2d1(x)
        #bs*64*1024*1
        x = self.conv2d2(x)
        #bs*128*1024*1
        x = self.conv2d3(x)
        #bs*1024*1024*1
        x, _ = torch.max(x, dim=2, keepdim=False)
        #bs*1024*1
        x = x.view(x.size(0), -1)
        #bs*1024
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #bs*9
        iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1)
        iden = iden.to(device='cuda') 
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x # bs*3*3



class adapt_layer_off(nn.Module):
    def __init__(self, num_node=64, offset_dim=3, trans_dim_in=64, trans_dim_out=64, fc_dim=64):
        super(adapt_layer_off, self).__init__()
        self.num_node = num_node
        self.offset_dim = offset_dim
        self.trans = conv_2d(trans_dim_in, trans_dim_out, 1)
        self.pred_offset = nn.Sequential(
            nn.Conv2d(trans_dim_out, offset_dim, kernel_size=1, bias=False),
            nn.Tanh())
        self.residual = conv_2d(trans_dim_in, fc_dim, 1)

    def forward(self, input_fea, input_loc):
        # Initialize node
        fpoint_idx = point_utils.farthest_point_sample(input_loc, self.num_node)  # (B, num_node)
        fpoint_loc = point_utils.index_points(input_loc, fpoint_idx)  # (B, 3, num_node)
        fpoint_fea = point_utils.index_points(input_fea, fpoint_idx)  # (B, C, num_node)
        group_idx = point_utils.query_ball_point(0.3, 64, input_loc, fpoint_loc)   # (B, num_node, 64)
        group_fea = point_utils.index_points(input_fea, group_idx)  # (B, C, num_node, 64)
        group_fea = group_fea - fpoint_fea.unsqueeze(3).expand(-1, -1, -1, self.num_node)

        # Learn node offset
        seman_trans = self.pred_offset(group_fea)  # (B, 3, num_node, 64)
        group_loc = point_utils.index_points(input_loc, group_idx)   # (B, 3, num_node, 64)
        group_loc = group_loc - fpoint_loc.unsqueeze(3).expand(-1, -1, -1, self.num_node)
        node_offset = (seman_trans*group_loc).mean(dim=-1)

        # Update node and get node feature
        node_loc = fpoint_loc+node_offset.squeeze(-1)  # (B,3,num_node)
        group_idx = point_utils.query_ball_point(None, 64, input_loc, node_loc)
        residual_fea = self.residual(input_fea)
        group_fea = point_utils.index_points(residual_fea, group_idx)
        node_fea, _ = torch.max(group_fea, dim=-1, keepdim=True)

        # Interpolated back to original point
        output_fea = point_utils.upsample_inter(input_loc, node_loc, input_fea, node_fea, k=3).unsqueeze(3)

        return output_fea, node_fea, node_offset

def hlr(logits):
    """Compute the hard likelihood loss ratio."""
    hlr_loss =0
    max_idxs = torch.argmax(logits, 1)
    for i,l in enumerate(logits):
        max_ind = max_idxs[i]
        new_l = torch.cat((l[:max_ind], l[max_ind+1:]))
        value = torch.logsumexp(new_l,0)
        loss= -l[max_ind] + value
        hlr_loss += loss
    return hlr_loss/len(logits)
