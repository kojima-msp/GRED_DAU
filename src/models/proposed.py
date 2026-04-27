import numpy as np
import torch
from torch import nn
from typing import Optional
import warnings
import tqdm

import sys
sys.path.append("../models")
from existing import LaplacianRegularization
from PnP import Graph_PnP

warnings.filterwarnings("ignore")

class Graph_RED(nn.Module):
    """
    """
    def __init__(self,layers:int=10,denoiser_name:str='lr', init_params:Optional[dict]=None, single_param:bool=False):
        """__init__
        Parameters
        ----------
        layers : int
            depth of layers
        denoiser_name : str
            name of denoiser
        """
        super().__init__()

        self.layers = layers
        self.denoiser_name = denoiser_name
        self.single_param = single_param

        self.set_params(init_params)

    def _nabla(self,matrix,y,ite):
        ite=0 if self.single_param==True else ite
        if(self.denoiser_name=='lr'):
            return matrix - y + torch.pow(10, self.alpha_red[ite])* (matrix - self.denoiser(matrix,self.L,self.alpha_lr[ite]))
            
        elif(self.denoiser_name=='pnp'):
            return matrix - y + torch.pow(10, self.alpha_red[ite])* (matrix - self.denoiser(matrix,self.L,self.alpha_pnp[ite],self.alpha_lr[ite]))

    def _matrix_product(self,matrix1,matrix2):
        return torch.sum(torch.mul(matrix1,matrix2))

    def forward(self,Y,L):
        """
        Parameters
        ----------
        Y : torch.Tensor (NxD)
            observed data
        L : torch.Tensor (NxN)
            graph Laplacian matrix

        Returns
        ----------
        x : torch.Tensor (NxD)
            denoised signal
        """
        self.Y = Y
        self.L = L

        MIN_NUM = 10**-15

        data_node_rec = torch.zeros_like(Y)

        self.denoiser.reset_params()
        
        for d in range(Y.shape[1]):

            # preprocessing
            y = Y[:,d]
            x = torch.zeros_like(y)
            nablax_prev = self._nabla(x,y,0)
            delta_x = -1*nablax_prev
            

            # iteration
            for ite in range(1, self.layers+1):

                # Stepsize decision
                tau = -1 * self._matrix_product(delta_x,nablax_prev)/(self._matrix_product(delta_x,self._nabla(delta_x,y,ite)+y) + MIN_NUM)
                # --------------------

                # Search direction updating
                x = x + tau*delta_x
                nablax = self._nabla(x, y, ite)
                gamma = self._matrix_product(nablax,nablax)/(self._matrix_product(nablax_prev,nablax_prev) + MIN_NUM)
                delta_x = -1*nablax + gamma * delta_x

                nablax_prev = nablax

            data_node_rec[:,d] = x
        
        return data_node_rec
    
    def set_params(self, init_params):
        """set_params
        Parameters
        ----------
        init_params : dict
            initial parameters for denoiser
        """
        # trainable params
        init_params_red = init_params['red'] if init_params is not None else 1.0
        self.alpha_red = nn.ParameterList([nn.Parameter(torch.tensor(init_params_red)) for _ in range(self.layers+1)])

        if(self.denoiser_name=='lr'):
            init_params_lr = init_params['lr'] if init_params is not None else 1.0
            self.denoiser = LaplacianRegularization()
            self.alpha_lr = nn.ParameterList([nn.Parameter(torch.tensor(init_params_lr)) for _ in range(self.layers+1)])

        elif(self.denoiser_name=='pnp'):
            init_params_lr = init_params['lr'] if init_params is not None else 1.0
            init_params_pnp = init_params['pnp'] if init_params is not None else 1.0

            self.denoiser = Graph_PnP(layers=10)
            self.alpha_pnp = nn.ParameterList([nn.Parameter(torch.tensor(init_params_pnp)) for _ in range(self.layers+1)])
            self.alpha_lr = nn.ParameterList([nn.Parameter(torch.tensor(init_params_lr)) for _ in range(self.layers+1)])    


class N2N_DAU_RED():
    def __init__(self,denoiser_name='lr',layers=10):
        """__init__
        Parameters
        ----------
        denoiser_name : str
            name of denoiser (default:lr)
        layers : int
            number of layers (default:100)
        """

        self.model = Graph_RED(layers=layers,denoiser_name=denoiser_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=.1)
        self.criterion = torch.nn.MSELoss()

    def forward(self,data_node_obs,L,noise_n2n=None,unsupervised_epochs=100):
        """forward
        Parameters
        ----------
        data_node_obs : Matrix(NxD)
            observed data
        L : Matrix(NxN)
            graph laplacian matrix
        noise_n2n : float or None
            noise level of Noise2Noise
            if None, noise_n2n is set to data_node_obs.max()*0.40
        unsupervised_epochs : int
            number of epochs (default:100)
        """
        N_trial = 5
        rng = np.random.default_rng()
        if noise_n2n is None:
            noise_n2n = data_node_obs.max()*0.40
        

        # train
        pbar = tqdm.tqdm(range(unsupervised_epochs))

        self.model.train()

        for epoch in pbar:
            
            loss = 0

            for k in range(N_trial):
                # add noise
                data_node_n2n = data_node_obs + torch.normal(mean=torch.zeros(data_node_obs.shape), std=rng.random()*noise_n2n)
            
                x_tilde = self.model.forward(data_node_n2n,L)
                loss += self.criterion(x_tilde,torch.tensor(data_node_obs))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({'loss': np.sqrt(loss.cpu().detach().numpy())/N_trial})

        self.model.eval()
        return self.model.forward(data_node_obs,L)