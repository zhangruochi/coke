import torch

def init_weights(layer):

    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, 1)
        torch.nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.normal_(layer.weight, 0, 0.01)
        torch.nn.init.constant_(layer.bias, 0)


def init_weights_2(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
#         torch.nn.init.normal_(layer.weight, 0.0, 0.1)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
    elif isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0) 