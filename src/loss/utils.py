import torch
from torch import nn
from torch.nn import Functional as F

def cross_entropy_loss(
        predict,
        label,
        weights
):
    assert predict.shape == label.shape
    