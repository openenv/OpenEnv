import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, calculates the mean, and then applies softmax.
    """

    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.softmax(x, dim=1)
        return x


batch_size = 128
in_features = 100
out_features = 50
dropout_p = 0.2


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, dropout_p]
