import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 16)
        self.conv3 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu((x), negative_slope=0.2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu((x), negative_slope=0.2)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        return x

def focal_loss(logits, labels, gamma=2, reduction="mean"):
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    log_pt = -ce_loss
    pt = torch.exp(log_pt)
    weights = (1 - pt) ** gamma
    fl = weights * ce_loss

    if reduction == "sum":
        fl = fl.sum()
    elif reduction == "mean":
        fl = fl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return fl

def balanced_focal_loss(logits, labels, alpha=1.0, gamma=2, reduction="mean"):
    return alpha * focal_loss(logits, labels, gamma, reduction)

