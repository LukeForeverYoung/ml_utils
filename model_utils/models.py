from torch import nn
from torch.nn.utils.weight_norm import weight_norm

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU',last_act=False, dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act and last_act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)