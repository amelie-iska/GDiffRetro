
from .common import MultiLayerPerceptron
from .conv import  GraphConv,  RelationalGraphConv 
from .readout import MeanReadout, SumReadout, MaxReadout, Softmax

MLP = MultiLayerPerceptron
GCNConv = GraphConv
RGCNConv = RelationalGraphConv

__all__ = [
    "MultiLayerPerceptron", 
    "GraphConv", "RelationalGraphConv",
    "MeanReadout", "SumReadout", "MaxReadout", "Softmax", 
    "MLP",  "GCNConv", "RGCNConv"
]