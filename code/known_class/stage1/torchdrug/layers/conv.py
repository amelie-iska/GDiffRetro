import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max

from torchdrug import data, layers, utils
from torchdrug.layers import functional


class MessagePassingBase(nn.Module):
    """
    Base module for message passing.

    Any custom message passing module should be derived from this class.
    """
    gradient_checkpoint = False

    def message(self, graph, input):
        """
        Compute edge messages for the graph.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: edge messages of shape :math:`(|E|, ...)`
        """
        raise NotImplementedError

    def aggregate(self, graph, message):
        """
        Aggregate edge messages to nodes.

        Parameters:
            graph (Graph): graph(s)
            message (Tensor): edge messages of shape :math:`(|E|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def message_and_aggregate(self, graph, input):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(graph, input)
        update = self.aggregate(graph, message)
        return update

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        input = tensors[-1]
        update = self.message_and_aggregate(graph, input)
        return update

    def combine(self, input, update):
        """
        Combine node input and node update.

        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def forward(self, graph, input):
        """
        Perform message passing over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update)
        return output


class GraphConv(MessagePassingBase):
    """
    Graph convolution operator from `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        message = input[node_in]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            message += edge_input
        message /= degree_in[node_in].sqrt()
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / degree_out.sqrt()
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in + 1
        degree_out = graph.degree_out + 1
        edge_weight = edge_weight / (degree_in[node_in] * degree_out[node_out]).sqrt()
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output




class RelationalGraphConv(MessagePassingBase):
    """
    Relational graph convolution operator from `Modeling Relational Data with Graph Convolutional Networks`_.

    .. _Modeling Relational Data with Graph Convolutional Networks:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(RelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message
    
    def aggregate(self, graph, message):

        assert graph.num_relation == self.num_relation

        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) / \
                 (scatter_add(edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) + self.eps)
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
        edge_weight = graph.edge_weight / degree_out[node_out]
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, input, update):
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output




