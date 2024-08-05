from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

EPS = 1e-15
MAX_LOGSTD = 10

class GCNEncoder(torch.nn.Module): # VANILLA GRAPH ENCODER CLASS
    def __init__(self, in_channels, out_channels): 
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # FIRST GRAPH CONVOLUTION LAYER
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # SECOND GRAPH CONVOLUTION LAYER

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu() # PASSING THROUGH GRAPH CONVOLUTION LAYERS
        return self.conv2(x, edge_index).relu()


class VariationalGCNEncoder(torch.nn.Module): # VARIATIONAL GRAPH ENCODER CLASS
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # FIRST GRAPH CONVOLUTION LAYER
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels, cached=True) # SECOND GRAPH CONVOLUTION LAYER
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True) # CONVOLUTION LAYER FOR MEAN OF GAUSSIAN DISTRIBUTION
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True) # CONVOLUTION LAYER FOR VARIANCE OF GAUSSIAN DISTRIBUTION

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu() # PASSING THROUGH GRAPH CONVOLUTION LAYERS
        x = self.conv2(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index) # RETURN THE PARAMETERS OF GAUSSIAN DISTRIBUTION



class InnerProductDecoder(torch.nn.Module): # DECODER CLASS FOR AUTOENCODER

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        sigmoid: bool = True,
    ) -> Tensor:

        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1) # PRODUCT OF THE TWO FEATURE EMBEDDINGS OF TWO NODES
        return torch.sigmoid(value) if sigmoid else value # SCALE THE SIMILARITY SCORE BETWEEN 0 AND 1

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:

        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module): # GRAPH AUTOENCODER CLASS

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder # REPRESENTING THE ENCODER CLASS
        self.decoder = InnerProductDecoder() if decoder is None else decoder # REPRESENTING THE DECODER CLASS
        GAE.reset_parameters(self)

    def reset_parameters(self):
        
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:
       
        return self.encoder(*args, **kwargs) # FORWARD PASS

    def encode(self, *args, **kwargs) -> Tensor:
        
        return self.encoder(*args, **kwargs) # OBTAIN ENCODER REPRESENTATION

    def decode(self, *args, **kwargs) -> Tensor:
       
        return self.decoder(*args, **kwargs) # OBTAIN DECODER REPRESENTATION

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean() # FIRST COMPONENT OF RECONSTRUCTION LOSS 

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()  # SECOND COMPONENT OF RECONSTRUCTION LOSS

        return pos_loss + neg_loss # RETURN ALL RECONSTRUCTION LOSSES


class VGAE(GAE): # VARIATIONAL GRAPH AUTOENCODER CLASS

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor: # REPARAMETERIZATION TRICK FOR BACKPROPAGATION OF GRADIENTS
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor: # ENCODING MMODULE
      
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs) # OBTAINING PARAMETERS OF GAUSSIAN DISTRIBUTION FROM ENCODER
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD) 
        z = self.reparametrize(self.__mu__, self.__logstd__) # PERFORM REPARAMETERIZATION TRICK FOR BAXKPROPAGATION OF GRADIENTS FOR VAE
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor: # COMPUTE KL DIVERGENCE LOSS

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))



