import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add, scatter_max


class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, graph.node2graph, dim=0, dim_size=graph.batch_size)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, graph.node2graph, dim=0, dim_size=graph.batch_size)
        return output


class MaxReadout(nn.Module):
    """Max readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        output = scatter_max(input, graph.node2graph, dim=0, dim_size=graph.batch_size)[0]
        return output


class Softmax(nn.Module):
    """Softmax operator over graphs with variadic sizes."""

    eps = 1e-10

    def forward(self, graph, input):
        """
        Perform softmax over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node logits

        Returns:
            Tensor: node probabilities
        """
        x = input - scatter_max(input, graph.node2graph, dim=0, dim_size=graph.batch_size)[0][graph.node2graph]
        x = x.exp()
        normalizer = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)[graph.node2graph]
        return x / (normalizer + self.eps)


