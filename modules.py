import torch
import torch.nn as nn


class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """
    def __init__(self, depth_of_mlp, in_features, out_features):
        super().__init__()

        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)

        self.skip = SkipConnection(in_features+out_features, out_features)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(in1=inputs, in2=mult)
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out


class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
