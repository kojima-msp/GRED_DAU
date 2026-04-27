import os
import sys
sys.path.append(os.path.dirname(__file__))
from existing import LaplacianRegularization
import torch
from torch import nn
import numpy as np

class Graph_PnP(nn.Module):
    def __init__(self,layers=10):
        """__init__
        Parameters
        ----------
        layers : int
            depth of layers (default:10)
        """
        super(Graph_PnP, self).__init__()
        self.layers = layers
        self.denoiser = LaplacianRegularization()
        
    def forward(self,y,L,alpha_pnp,alpha_lr):

        x = y
        s = torch.zeros(y.shape)
        t = torch.zeros(y.shape)

        # iteration
        for k in range(self.layers):

            x = 1/(1+ torch.pow(10, alpha_pnp)) * (y + torch.pow(10, alpha_pnp)*(s-t))
            s = self.denoiser.forward(x+t,L,alpha_lr)
            t = t+(x-s)

        return x
    
    def reset_params(self):
        self.denoiser.reset_params()