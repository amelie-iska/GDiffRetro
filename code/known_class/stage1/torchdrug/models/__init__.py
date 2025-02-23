from .gcn import GraphConvolutionalNetwork, RelationalGraphConvolutionalNetwork

# alias
GCN = GraphConvolutionalNetwork
RGCN = RelationalGraphConvolutionalNetwork

__all__ = [
     "GraphConvolutionalNetwork", "RelationalGraphConvolutionalNetwork",
     "GCN", "RGCN"
]