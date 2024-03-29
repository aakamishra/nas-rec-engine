import torch
from utils.model_utils import device
from model_src.model_components import GraphAggregator
import numpy as np
import sklearn.mixture as mixture

class CGNodeEmbedding(torch.nn.Module):

    def __init__(self, n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                 bias_embed_size=2, n_unique_kernels=8, n_shape_vals=6):
        super(CGNodeEmbedding, self).__init__()
        assert kernel_embed_size % 2 == 0
        self.out_embed_size = out_embed_size
        self.n_unique_labels = n_unique_labels
        self.n_unique_kernels = n_unique_kernels
        self.n_shape_vals = n_shape_vals
        regular_out_embed_size = out_embed_size - shape_embed_size
        weighted_out_embed_size = out_embed_size - shape_embed_size - kernel_embed_size - bias_embed_size
        self.regular_embed_layer = torch.nn.Embedding(n_unique_labels, regular_out_embed_size)
        self.weighted_embed_layer = torch.nn.Embedding(n_unique_labels, weighted_out_embed_size)
        self.kernel_embed_layer = torch.nn.Embedding(n_unique_kernels, kernel_embed_size // 2)
        self.shape_embed_layer = torch.nn.Linear(n_shape_vals, shape_embed_size)
        self.bias_embed_layer = torch.nn.Linear(2, bias_embed_size)

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias):
        regular_embedding = None
        weighted_embedding = None
        if regular_node_inds is not None:
            regular_embedding = self.regular_embed_layer(regular_node_inds.to(device()))
            shape_embedding = self.shape_embed_layer(regular_node_shapes.to(device()))
            regular_embedding = torch.cat([regular_embedding,
                                           shape_embedding], dim=-1)
        if weighted_node_inds is not None:
            weighted_embedding = self.weighted_embed_layer(weighted_node_inds.to(device()))
            kernel_embedding = self.kernel_embed_layer(weighted_node_kernels.to(device()))
            kernel_embedding = kernel_embedding.view(kernel_embedding.shape[0],
                                                     kernel_embedding.shape[1], -1)
            shape_embedding = self.shape_embed_layer(weighted_node_shapes.to(device()))
            bias_embedding = self.bias_embed_layer(weighted_node_bias.to(device()))
            weighted_embedding = torch.cat([weighted_embedding,
                                            shape_embedding,
                                            kernel_embedding,
                                            bias_embedding], dim=-1)
        if regular_embedding is not None and weighted_embedding is not None:
            # Implicit rule of weighted node always in front, very important
            node_embedding = torch.cat([weighted_embedding, regular_embedding], dim=1)
        elif regular_embedding is not None:
            node_embedding = regular_embedding
        elif weighted_embedding is not None:
            node_embedding = weighted_embedding
        else:
            raise ValueError("Input to CGNodeEmbedding cannot both be None")
        return node_embedding


class CGRegressor(torch.nn.Module):

    def __init__(self, embed_layer, encoder, aggregator, hidden_size,
                 activ=None, ext_feat_size=0):
        super(CGRegressor, self).__init__()
        self.embed_layer = embed_layer
        self.encoder = encoder
        self.aggregator = aggregator
        self.activ = activ
        self.post_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.regressor = torch.nn.Linear(hidden_size + ext_feat_size, 1)

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                edge_tsr_list, batch_last_node_idx_list, index=None,
                ext_feat=None):
        node_embedding = self.embed_layer(regular_node_inds, regular_node_shapes,
                                          weighted_node_inds, weighted_node_shapes,
                                          weighted_node_kernels, weighted_node_bias)
        batch_embedding = self.encoder(node_embedding, edge_tsr_list, batch_last_node_idx_list)
        graph_embedding = self.aggregator(batch_embedding, batch_last_node_idx_list, index=index)
        graph_embedding = self.post_proj(graph_embedding)
        if self.activ is not None:
            graph_embedding = self.activ(graph_embedding)
        if ext_feat is not None:
            ext_feat = ext_feat.to(device())
            graph_embedding = torch.cat([graph_embedding, ext_feat], dim=-1)
        out = self.regressor(graph_embedding)
        return out


def make_cg_regressor(n_unique_labels, out_embed_size,
                      shape_embed_size, kernel_embed_size,
                      n_unique_kernels, n_shape_vals,
                      hidden_size, out_channels, gnn_constructor,
                      bias_embed_size=2, gnn_activ=torch.nn.ReLU(),
                      n_gnn_layers=4, dropout_prob=0.0,
                      regressor_activ=None, aggr_method="mean"):
    from model_src.model_components import PreEmbeddedGraphEncoder
    embed_layer = CGNodeEmbedding(n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                                  bias_embed_size, n_unique_kernels, n_shape_vals)
    encoder = PreEmbeddedGraphEncoder(out_embed_size, hidden_size, out_channels, gnn_constructor,
                                      gnn_activ, n_gnn_layers, dropout_prob)
    aggregator = GraphAggregator(hidden_size, aggr_method=aggr_method)
    regressor = CGRegressor(embed_layer, encoder, aggregator, hidden_size, activ=regressor_activ)
    
    return regressor

class GraphDecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z, num_nodes):
        # z is the latent representation generated by the encoder
        # num_nodes is the number of nodes in the original graph
        h = torch.relu(self.fc1(z))
        h = self.fc2(h)
        # Reshape the output to match the number of nodes in the original graph
        output_shape = (-1, num_nodes, self.output_dim)
        x_hat = torch.reshape(h, output_shape)
        # Apply a softmax activation to obtain the reconstructed graph
        x_hat = torch.softmax(x_hat, dim=2)
        return x_hat
    

class ClusterSpecificNet(torch.nn.Module):
    def __init__(self, num_clusters, input_size, hidden_size, output_size):
        super(ClusterSpecificNet, self).__init__()

        # Define the weights for each cluster
        self.cluster_weights = torch.nn.Parameter(torch.randn(num_clusters, input_size, hidden_size))
        self.cluster_bias = torch.nn.Parameter(torch.randn(num_clusters, hidden_size))
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2*hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        
        self.regressor = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, soft_label):
        
        w = torch.sum(soft_label.unsqueeze(-1).unsqueeze(-1) * self.cluster_weights, dim=0)
        b = torch.sum(soft_label.unsqueeze(-1) * self.cluster_bias, dim=0)

        # Compute the output using the weighted sum of the cluster weights and biases
        x = torch.nn.functional.relu(torch.matmul(x.double(), w.permute(2,1,0)).permute(1, 2, 0) + b)
        x = self.proj(torch.mean(x, dim=1).float())
        return self.regressor(x)
    
class GraphAutoEncoder(torch.nn.Module):

    def __init__(self, embed_layer, encoder, decoder, reduced_emebdding_sz=64):
        super(GraphAutoEncoder, self).__init__()
        self.embed_layer = embed_layer
        self.encoder = encoder
        self.projection_layer = torch.nn.Linear(2*self.encoder.hidden_size, reduced_emebdding_sz, bias=False)
        self.decoder = decoder
    
    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                edge_tsr_list, batch_last_node_idx_list, index=None,
                ext_feat=None):
        node_embedding = self.embed_layer(regular_node_inds, regular_node_shapes,
                                          weighted_node_inds, weighted_node_shapes,
                                          weighted_node_kernels, weighted_node_bias)
        
        encoder_output = self.encoder(node_embedding, edge_tsr_list, batch_last_node_idx_list)
        
        batch_embedding1 = self.projection_layer(encoder_output)
        
        batch_embedding2 = self.projection_layer(self.encoder(node_embedding, edge_tsr_list, batch_last_node_idx_list))
        
        decoded_node_embedding = self.decoder(encoder_output, node_embedding.shape[1])
                        
        return torch.cat([batch_embedding1, batch_embedding2], dim=0), node_embedding, decoded_node_embedding
    
def make_embedding_model(n_unique_labels, out_embed_size,
                      shape_embed_size, kernel_embed_size,
                      n_unique_kernels, n_shape_vals,
                      hidden_size, out_channels, gnn_constructor,
                      bias_embed_size=2, gnn_activ=torch.nn.ReLU(),
                      n_gnn_layers=4, dropout_prob=0.0,
                      regressor_activ=None, aggr_method="mean"):
    from model_src.model_components import PreEmbeddedGraphEncoderWithAttention
    embed_layer = CGNodeEmbedding(n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                                  bias_embed_size, n_unique_kernels, n_shape_vals)
    encoder = PreEmbeddedGraphEncoderWithAttention(out_embed_size, hidden_size, out_channels, gnn_constructor,
                                      gnn_activ, n_gnn_layers, dropout_prob)    
    # decoder = TransformerDecoder(output_size=out_embed_size, 
    #                              hidden_size=2*hidden_size, 
    #                              num_layers=3, 
    #                              num_heads=4, 
    #                              dropout=0.2)
    decoder = GraphDecoder(2*encoder.hidden_size, 4*encoder.hidden_size, encoder.hidden_size)
    
    return GraphAutoEncoder(embed_layer, encoder, decoder)


class AggregateCGRegressor(torch.nn.Module):

    def __init__(self, graph_encoder, aggregator, hidden_size,
                 activ=None, ext_feat_size=0, num_heads=4):
        super(AggregateCGRegressor, self).__init__()
        self.graph_encoder = graph_encoder
        self.aggregator = aggregator
        self.activ = activ
        self.post_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        
        self.regressor = torch.nn.Linear(hidden_size + ext_feat_size, 1)

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                edge_tsr_list, batch_last_node_idx_list, index=None,
                ext_feat=None, mask=None):
        node_embedding = self.graph_encoder.embed_layer(regular_node_inds, regular_node_shapes,
                                          weighted_node_inds, weighted_node_shapes,
                                          weighted_node_kernels, weighted_node_bias)
        batch_embedding = self.graph_encoder.encoder(node_embedding, edge_tsr_list, batch_last_node_idx_list)
        
        graph_embedding = self.aggregator(batch_embedding, batch_last_node_idx_list, index=index)
        
        out = self.regressor(graph_embedding)
        return out

def make_embedding_regressor_model(n_unique_labels, out_embed_size,
                      shape_embed_size, kernel_embed_size,
                      n_unique_kernels, n_shape_vals,
                      hidden_size, out_channels, gnn_constructor,
                      bias_embed_size=2, gnn_activ=torch.nn.ReLU(),
                      n_gnn_layers=4, dropout_prob=0.0,
                      regressor_activ=None, aggr_method="mean"):
    from model_src.model_components import PreEmbeddedGraphEncoderWithAttention
    
    embed_layer = CGNodeEmbedding(n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                                  bias_embed_size, n_unique_kernels, n_shape_vals)
    encoder = PreEmbeddedGraphEncoderWithAttention(out_embed_size, hidden_size, out_channels, gnn_constructor,
                                      gnn_activ, n_gnn_layers, dropout_prob)   
    
    decoder = GraphDecoder(2*encoder.hidden_size, 4*encoder.hidden_size, encoder.hidden_size)
    
    graph_encoder = GraphAutoEncoder(embed_layer, encoder, decoder)
    
    aggregator = GraphAggregator(2*hidden_size, aggr_method=aggr_method)
    
    return AggregateCGRegressor(graph_encoder, aggregator, 2*hidden_size, activ=regressor_activ)


def save_gmm(gmm, name):
    gmm_name = name
    folder_path = "/home/ec2-user/nas-rec-engine/"
    np.save(folder_path + gmm_name + '_weights', gmm.weights_, allow_pickle=False)
    np.save(folder_path + gmm_name + '_means', gmm.means_, allow_pickle=False)
    np.save(folder_path + gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)
    print("GMM Saved", name)

# reload
def load_gmm(gmm_name):
    folder_path = "/home/ec2-user/nas-rec-engine/"
    means = np.load(folder_path + gmm_name + '_means.npy')
    covar = np.load(folder_path + gmm_name + '_covariances.npy')
    loaded_gmm = mixture.GaussianMixture(n_components = len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(folder_path + gmm_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    
    return loaded_gmm



class AggregateCGRegressorV2(torch.nn.Module):

    def __init__(self, graph_encoder, aggregator, hidden_size,
                 activ=None, ext_feat_size=0, num_heads=4, num_clusters=20):
        super(AggregateCGRegressorV2, self).__init__()
        self.graph_encoder = graph_encoder
        self.aggregator = aggregator
        self.activ = activ
        
        self.gmm = load_gmm("nas_assignment_model1")
        
        self.regressor = ClusterSpecificNet(32, hidden_size, hidden_size=hidden_size*2, output_size=hidden_size)
        
        self.i = 0

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                edge_tsr_list, batch_last_node_idx_list, index=None,
                ext_feat=None, mask=None):
        node_embedding = self.graph_encoder.embed_layer(regular_node_inds, regular_node_shapes,
                                          weighted_node_inds, weighted_node_shapes,
                                          weighted_node_kernels, weighted_node_bias)
        batch_embedding = self.graph_encoder.encoder(node_embedding, edge_tsr_list, batch_last_node_idx_list)
        
        graph_embedding = self.aggregator(batch_embedding, batch_last_node_idx_list, index=index)
        
        #gmm_training_path = "/home/ec2-user/nas-rec-engine/gmm_training/"
        #np.save(gmm_training_path + f"{self.i}.npy", graph_embedding.cpu().detach().numpy())
        
        cluster_label = torch.from_numpy(self.gmm.predict_proba(graph_embedding.cpu().detach().numpy())).to(device())
        
        out = self.regressor(graph_embedding, cluster_label)
        
        return out
    

def make_embedding_regressor_modelV2(n_unique_labels, out_embed_size,
                      shape_embed_size, kernel_embed_size,
                      n_unique_kernels, n_shape_vals,
                      hidden_size, out_channels, gnn_constructor,
                      bias_embed_size=2, gnn_activ=torch.nn.ReLU(),
                      n_gnn_layers=4, dropout_prob=0.0,
                      regressor_activ=None, aggr_method="mean"):
    from model_src.model_components import PreEmbeddedGraphEncoderWithAttention
    
    embed_layer = CGNodeEmbedding(n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                                  bias_embed_size, n_unique_kernels, n_shape_vals)
    encoder = PreEmbeddedGraphEncoderWithAttention(out_embed_size, hidden_size, out_channels, gnn_constructor,
                                      gnn_activ, n_gnn_layers, dropout_prob)   
    
    decoder = GraphDecoder(2*encoder.hidden_size, 4*encoder.hidden_size, encoder.hidden_size)
    
    graph_encoder = GraphAutoEncoder(embed_layer, encoder, decoder)
    
    aggregator = GraphAggregator(2*hidden_size, aggr_method=aggr_method)
    
    return AggregateCGRegressorV2(graph_encoder, aggregator, 2*hidden_size, activ=regressor_activ)
