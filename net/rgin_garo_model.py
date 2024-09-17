import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
import math
from torch_sparse import spspmm
import numpy as np
import torch_geometric.utils as pyg_utils
from torch.nn import Parameter
from net.brainmsgpassing import MyMessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch.nn import Parameter
from net.brainmsgpassing import MyMessagePassing
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch.nn import Parameter
from net.brainmsgpassing import MyMessagePassing
# from net.brainmsgpassing import MyMessagePassing
from torch_geometric.utils import add_remaining_self_loops,softmax

from torch_geometric.typing import (OptTensor)

class MyGINConvWithMean(MyMessagePassing):
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=False, eps=0.0,
                 **kwargs):
        super(MyGINConvWithMean, self).__init__(aggr='add', **kwargs)  # Use 'add' for GIN

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.epsilon = Parameter(torch.Tensor(1).fill_(eps))  # Learnable epsilon

        # Existing nn for initial weight transformation
        self.nn = nn
        
        mlp_update = torch.nn.Sequential(torch.nn.Linear(self.out_channels, self.out_channels * 2),
                                        # torch.nn.BatchNorm1d(self.out_channels * 2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.out_channels * 2, self.out_channels))
        self.mlp_update = mlp_update

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()
        self.epsilon.data.fill_(0.0)  # Reset epsilon to initial value

    def forward(self, x, edge_index, edge_weight=None, pseudo=None, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(0))

        # Apply the nn to transform input features
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)

        # Transform node features with the weights
        if torch.is_tensor(x):
            x_transformed = torch.matmul(x.unsqueeze(1), weight).squeeze(1)  # W_i * h_i
        else:
            x_transformed = (
                None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
                None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1)
            )
        
        # Apply message passing
        aggr_out = self.propagate(edge_index, size=size, x=x_transformed, edge_weight=edge_weight)

        # Combine with transformed node features and apply MLP
        updated_features = (1 + self.epsilon) * x_transformed + aggr_out
        out = self.mlp_update(updated_features)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out


    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
        # Apply edge weight normalization if provided
        if edge_weight is not None:
            edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        # Apply the learnable bias
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        
        # MLP is applied in the forward function
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, epsilon={:.4f})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.epsilon.item()
        )

##########################################################################################################################
class CustomNetworkWithGARO(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, n_hidden_layers, n_fc_layers, k, R=100, reg= 0.1):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(CustomNetworkWithGARO, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.n_fc_layers = n_fc_layers
        self.indim = indim
        self.ratio = ratio
        self.reg = reg

        self.k = k
        self.R = R

        self.allnns = nn.ModuleList()
        self.allconvs = nn.ModuleList()
        self.allpools = nn.ModuleList()
        self.gmts = nn.ModuleList()

        #Fully Connected Layers
        self.allfcs = nn.ModuleList()
        self.batchnorms= nn.ModuleList()

        self.pnaconvs = nn.ModuleList()
        self.garos = nn.ModuleList()

        #Graph Convolution and Pooling
        

        for i in range(len(n_hidden_layers)):
            if(i==0):
                self.allnns.append(nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, n_hidden_layers[i] * self.indim)))
                self.allconvs.append(MyGINConvWithMean(self.indim, n_hidden_layers[i], self.allnns[i], normalize=False))
                self.allpools.append(TopKPooling(n_hidden_layers[i], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid))
                self.garos.append(ModuleGARO(output_dim=n_hidden_layers[i] , hidden_dim=n_hidden_layers[i] , dropout=0.1, upscale=1.0))
            else:
                self.allnns.append(nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, n_hidden_layers[i] * n_hidden_layers[i-1])))
                self.allconvs.append(MyGINConvWithMean(n_hidden_layers[i-1], n_hidden_layers[i], self.allnns[i], normalize=False))
                self.allpools.append(TopKPooling(n_hidden_layers[i], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid))
                #self.gmts.append(GraphMultisetTransformer(n_hidden_layers[i] , 128, n_hidden_layers[i] , num_nodes=R))
                self.garos.append(ModuleGARO(output_dim=n_hidden_layers[i] , hidden_dim=n_hidden_layers[i] , dropout=0.1, upscale=1.0))


        for i in range(len(n_fc_layers)):
            if i==0:
                final_layer_output = sum(2*int(x) for x in n_hidden_layers)
                final_conv_layer = n_hidden_layers[-1]
                self.allfcs.append(torch.nn.Linear((np.sum(n_hidden_layers) ), n_fc_layers[i]))
                self.batchnorms.append(torch.nn.BatchNorm1d(n_fc_layers[i]))
            else:
                self.allfcs.append(torch.nn.Linear(n_fc_layers[i-1], n_fc_layers[i]))
                self.batchnorms.append(torch.nn.BatchNorm1d(n_fc_layers[i]))
        
        

        self.finallayer = torch.nn.Linear(n_fc_layers[len(n_fc_layers)-1], nclass)

    def forward(self, x, edge_index, batch, edge_attr, pos):

        #Graph Convolution Part
        scores=[]
        garos=[]
        batch_size = x.shape[0]//self.R

        for i in range(len(self.n_hidden_layers)):

            x = self.allconvs[i](x, edge_index, edge_attr, pos)            
            x, edge_index, edge_attr, batch, perm, score = self.allpools[i](x, edge_index, edge_attr, batch)
            pos = pos[perm]

            garo_output,_ = self.garos[i](x, batch_size)
            scores.append(score)

            edge_attr = edge_attr.squeeze()
            edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
            garos.append(garo_output)


        
        #Fully Connected Layer Part

        for i in range(len(self.n_fc_layers)):
            if i==0:
                x = torch.concat(garos, dim=1)
                x = self.batchnorms[i](F.relu(self.allfcs[i](x)))
                # x = F.relu(self.allfcs[i](x))
                x = F.dropout(x, p=self.ratio, training=self.training)
            else:
                x = self.batchnorms[i](F.relu(self.allfcs[i](x)))
                # x =F.relu(self.allfcs[i](x))
                x= F.dropout(x, p=self.ratio, training=self.training)
        
        x = torch.relu(self.finallayer(x))

        return x,self.allpools, scores


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight



class ModuleSERO(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, batch, node_axis=1):
        # assumes shape [... x node x ... x feature]
        #x = x.reshape(64,-1,x.shape[-1])
        x_readout = gap(x, batch)
        #x_readout = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(1))).mean(node_axis), x_graphattention
    

class ModuleGARO(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, batch_size, node_axis=-2):
        # assumes shape [... x node x ... x feature]
        x = x.reshape(batch_size,-1,x.shape[-1])
        x_q = self.embed_query(x.mean(node_axis))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q.unsqueeze(1), x_k.transpose(2,1))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.transpose(2,1))).mean(node_axis), x_graphattention
