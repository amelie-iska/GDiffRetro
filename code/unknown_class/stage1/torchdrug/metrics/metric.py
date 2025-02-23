import torch
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors

from torchdrug import utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.metrics.rdkit import sascorer

   

@R.register("metrics.accuracy")
def accuracy(pred, target):
    """
    Compute classification accuracy over sets with equal size.

    Suppose there are :math:`N` sets and :math:`C` categories.

    Parameters:
        pred (Tensor): prediction of shape :math:`(N, C)`
        target (Tensor): target of shape :math:`(N,)`
    """
    return (pred.argmax(dim=-1) == target).float().mean()



@R.register("metrics.variadic_accuracy")
def variadic_accuracy(input, target, size):
    """
    Compute classification accuracy over variadic sizes of categories.

    Suppose there are :math:`N` samples, and the number of categories in all samples is summed to :math:`B`.

    Parameters:
        input (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(N,)`. Each target is a relative index in a sample.
        size (Tensor): number of categories of shape :math:`(N,)`
    """
    index2graph = functional._size_to_index(size)

    input_class = scatter_max(input, index2graph)[1]
    target_index = target + size.cumsum(0) - size
    accuracy = (input_class == target_index).float()
    return accuracy
