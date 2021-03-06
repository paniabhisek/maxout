import torch
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def num_corrects(outputs, label_batch):
    """
    How many number of outputs of a model is
    equal to true labels

    :param outputs: Outputs of a model
    :type outputs: :py:class:`torch.Tensor`
    :param label_batch: True labels of a model
    :type label_batch: :py:class:`torch.Tensor`
    """
    out = outputs.argmax(1)
    corrects = out == label_batch
    return torch.sum(corrects).item()

def init_hyper_params():
    """
    Initialize hyper parameters.

    Stores number of neurons in each layer
    and number of layers before max operation for mlp
    Stores number of channels and kernel for
    each maxout layer
    """
    with open('maxout.json', 'r') as f:
        return json.load(f)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return (a * b) // gcd(a, b)
