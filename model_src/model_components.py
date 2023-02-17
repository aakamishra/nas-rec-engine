import torch
import numpy as np
from utils.model_utils import torch_geo_batch_to_data_list, device
from torch_geometric.utils.dropout import dropout_edge

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
except ModuleNotFoundError:
    print("Did not find torch_geometric, GNNs will be unavailable")


class GraphAggregator(torch.nn.Module):

    def __init__(self, hidden_size, aggr_method="last",
                 gnn_activ=None):
        super(GraphAggregator, self).__init__()
        self.aggr_method = aggr_method
        self.gnn_aggr_layer = PreEmbeddedGraphEncoder(hidden_size, hidden_size, hidden_size,
                                                      torch_geometric.nn.GraphConv,
                                                      activ=gnn_activ, n_layers=1)

    @staticmethod
    def _get_gnn_aggr_tsr_list(batch_size, n_nodes):
        src_list = [i for i in range(n_nodes - 1)]
        dst_list = [n_nodes - 1] * (n_nodes - 1)
        return [torch.LongTensor([src_list, dst_list]) for _ in range(batch_size)]

    def forward(self, node_embedding, batch_last_node_idx_list, index=None):
        assert len(node_embedding.shape) == 3
        if self.aggr_method == "sum":
            graph_embedding = node_embedding.sum(dim=1)
        elif self.aggr_method == "last":
            graph_embedding = node_embedding[:, -1, :]
        elif self.aggr_method == "mean":
            graph_embedding = node_embedding.mean(dim=1)
        elif self.aggr_method == "gnn":
            aggr_edge_tsr_list = self._get_gnn_aggr_tsr_list(node_embedding.shape[0], node_embedding.shape[1])
            node_embedding = self.gnn_aggr_layer(node_embedding, aggr_edge_tsr_list, batch_last_node_idx_list)
            graph_embedding = node_embedding[:, -1, :]
        elif self.aggr_method == "indexed":
            index = index.to(device())
            index = index.reshape(-1, 1, 1).repeat(1, 1, node_embedding.size(-1))
            graph_embedding = torch.gather(node_embedding, 1, index).squeeze(1)
        elif self.aggr_method == "none":
            graph_embedding = node_embedding
        elif self.aggr_method == "squeeze":
            assert len(node_embedding.shape) == 3 and node_embedding.shape[1] == 1, \
                "Invalid input shape: {}".format(node_embedding.shape)
            graph_embedding = node_embedding.squeeze(1)
        elif self.aggr_method == "flat":
            graph_embedding = node_embedding.reshape(node_embedding.shape[0], -1)
        elif self.aggr_method == "de-batch":
            graph_embedding = node_embedding.reshape(-1, node_embedding.shape[-1])
        else:
            raise ValueError("Unknown aggr_method: {}".format(self.aggr_method))
        return graph_embedding


class PreEmbeddedGraphEncoder(torch.nn.Module):

    def __init__(self, in_channels, hidden_size, out_channels, gnn_constructor,
                 activ=torch.nn.Tanh(), n_layers=4, dropout_prob=0.0,
                 add_normal_prior=False):
        super(PreEmbeddedGraphEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            input_size, output_size = hidden_size, hidden_size
            if i == 0:
                input_size = in_channels
            if i == n_layers - 1:
                output_size = out_channels
            gnn_layer = gnn_constructor(input_size, output_size)
            self.gnn_layers.append(gnn_layer)
        self.activ = activ
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.init_ff = torch.nn.Linear(2 * in_channels, in_channels)
        self.add_normal_prior = add_normal_prior

    def add_prior(self, embedding):
        prior = np.random.normal(size=embedding.shape)
        prior = torch.from_numpy(prior).float().to(device())
        embedding = torch.cat([embedding, prior], dim=-1)
        return self.init_ff(embedding)

    def forward(self, batch_node_tsr, edge_tsr_list, batch_last_node_idx_list):
        node_embedding = batch_node_tsr.to(device())
        if self.add_normal_prior and node_embedding.shape[1] == 1:
            node_embedding = self.add_prior(node_embedding)
        data_list = [Data(x=node_embedding[i,:], edge_index=edge_tsr_list[i].to(device()), edge_attr=None)
                     for i in range(node_embedding.shape[0])]
        torch_geo_batch = Batch.from_data_list(data_list)
        edge_index_tsr = torch_geo_batch.edge_index
        curr_gnn_output = torch_geo_batch.x
        for li, gnn_layer in enumerate(self.gnn_layers):
            curr_gnn_output = gnn_layer(curr_gnn_output, edge_index_tsr)
            if self.activ is not None:
                curr_gnn_output = self.activ(curr_gnn_output)
        curr_gnn_output = self.dropout(curr_gnn_output)
        batch_embedding_list = torch_geo_batch_to_data_list(curr_gnn_output, batch_last_node_idx_list,
                                                            batch_indicator=torch_geo_batch.batch)
        batch_embedding = torch.cat([t.unsqueeze(0) for t in batch_embedding_list], dim=0)
        return batch_embedding
    
    
class DoubleHeadedAttention(torch.nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.query_layer = torch.nn.Linear(input_dim, attention_dim)
        self.key_layer = torch.nn.Linear(input_dim, attention_dim)
        self.value_layer = torch.nn.Linear(input_dim, attention_dim)
        self.output_layer = torch.nn.Linear(attention_dim, input_dim)

    def forward(self, input1, input2):
        query1 = self.query_layer(input1)
        query2 = self.query_layer(input2)
        key1 = self.key_layer(input1)
        key2 = self.key_layer(input2)
        value1 = self.value_layer(input1)
        value2 = self.value_layer(input2)

        attention1 = torch.mm(query1, key2.transpose(0, 1))
        attention1 = torch.softmax(attention1, dim=-1)
        attention2 = torch.mm(query2, key1.transpose(0, 1))
        attention2 = torch.softmax(attention2, dim=-1)

        attended_input1 = torch.mm(attention1, value2)
        attended_input2 = torch.mm(attention2, value1)

        output1 = self.output_layer(attended_input1)
        output2 = self.output_layer(attended_input2)

        return output1, output2


class PreEmbeddedGraphEncoderWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, gnn_constructor,
                 activ=torch.nn.Tanh(), n_layers=4, dropout_prob=0.0,
                 add_normal_prior=False):
        super(PreEmbeddedGraphEncoderWithAttention, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            input_size, output_size = hidden_size, hidden_size
            if i == 0:
                input_size = in_channels
            if i == n_layers - 1:
                output_size = out_channels
            gnn_layer = gnn_constructor(input_size, output_size)
            self.gnn_layers.append(gnn_layer)
        self.activ = activ
        self.dropout_prob = dropout_prob
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.dropout_attn = torch.nn.Dropout(dropout_prob/2)
        self.init_ff = torch.nn.Linear(2 * in_channels, in_channels)
        self.add_normal_prior = add_normal_prior
        
        self.att_mechanism = DoubleHeadedAttention(hidden_size, hidden_size)
        self.combination_layer = torch.nn.Linear(2*hidden_size, hidden_size)


    def add_prior(self, embedding):
        prior = np.random.normal(size=embedding.shape)
        prior = torch.from_numpy(prior).float().to(device())
        embedding = torch.cat([embedding, prior], dim=-1)
        return self.init_ff(embedding)

    def forward(self, batch_node_tsr, edge_tsr_list, batch_last_node_idx_list):
        node_embedding = batch_node_tsr.to(device())
        node_embedding = torch.nn.functional.dropout(node_embedding, p=self.dropout_prob, training=self.training)
        if self.add_normal_prior and node_embedding.shape[1] == 1:
            node_embedding = self.add_prior(node_embedding)
        data_list = [Data(x=node_embedding[i,:], edge_index=dropout_edge(edge_tsr_list[i].to(device()), p=self.dropout_prob/2)[0], edge_attr=None)
                     for i in range(node_embedding.shape[0])]
        torch_geo_batch = Batch.from_data_list(data_list)
        edge_index_tsr = torch_geo_batch.edge_index
        curr_gnn_output = torch_geo_batch.x
        for li, gnn_layer in enumerate(self.gnn_layers):
            curr_gnn_output = gnn_layer(curr_gnn_output, edge_index_tsr)
            if self.activ is not None:
                curr_gnn_output = self.activ(curr_gnn_output)
        curr_gnn_output = self.dropout_attn(curr_gnn_output)
        batch_embedding_list = torch_geo_batch_to_data_list(curr_gnn_output, batch_last_node_idx_list,
                                                            batch_indicator=torch_geo_batch.batch)
        batch_embedding = torch.cat([t.unsqueeze(0) for t in batch_embedding_list], dim=0)
        
        attended_input1, attended_input2 = self.att_mechanism(torch_geo_batch.x, torch_geo_batch.x)
        combined_input = torch.cat([attended_input1, attended_input2], dim=-1)
        att_output = self.dropout_attn(self.combination_layer(combined_input))
        
        return torch.cat([batch_embedding, att_output.reshape(batch_embedding.shape)], dim=-1)

class PreEmbeddedGraphDecoder(torch.nn.Module):
    def __init__(self, hidden_size, edge_tsr_size, dropout):
        super(PreEmbeddedGraphEncoder, self).__init__()
        self.in_channels = edge_tsr_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.linear1(hidden_size, 2*hidden_size)
        self.linear2(2*hidden_size, 2*hidden_size)
        self.linear3(hidden_size, edge_tsr_size)
        
        self.linear4(hidden_size, edge_tsr_size)

        
    def forward(self, embedding):
        embedding = torch.F.dropout(embedding, p=self.dropout, training=self.training)
        x = torch.nn.ReLU(self.linear1(embedding))
        x = torch.nn.ReLU(self.linear2(x))
        x = torch.nn.ReLU(self.linear3(x))
        adj = torch.nn.ReLU(self.linear3(embedding))
        
        return x, adj
        
    
    
