import torch
from torch import nn, optim
from typing import Optional
import numpy as np
import torch_geometric as pyg
import copy

class LaplacianRegularization(nn.Module):
    def __init__(self):

        super().__init__()
        self.U = None
        self._lambda = None
    
    def forward(self,y,L,alpha):
        """forward
        Parameters
        ----------
        y : torch.Tensor
            input vector
        L : torch.Tensor
            Laplacian matrix
        alpha : torch.Tensor
        """
        if(self.U is None or self._lambda is None):
            self._lambda, self.U = torch.linalg.eigh(L)

        return self.U @ torch.diag(1/(1+torch.pow(10, alpha)*self._lambda)) @ self.U.T @ y
    
    def reset_params(self):
        self.U = None
        self._lambda = None

class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()
        self.eig_vals = None

    def forward(self, signal, A, alpha):

        if self.eig_vals is None:
            eig_vals, _ = torch.linalg.eigh(A)
            A = A/torch.max(torch.abs(eig_vals))

        Eye = torch.eye(signal.shape[0])
        A_hat = Eye - A
        H_A = torch.linalg.inv(Eye+alpha*A_hat.T @ A_hat)
        
        return H_A @ signal

class GCNN(nn.Module):
    def __init__(self, fts, input_fts, n_convs=3, n_lin=0, act=nn.ReLU(),
                 last_act=None, last_fts=1):
        super(GCNN, self).__init__()

        self.act = act
        self.last_act = last_act
        self.convs = nn.ModuleList()
        self.convs.append(pyg.nn.GCNConv(input_fts, fts))
        for i in range(1, n_convs):
            if (i == n_convs-1) and n_lin == 0:
                self.convs.append(pyg.nn.GCNConv(fts, last_fts))
            else:
                self.convs.append(pyg.nn.GCNConv(fts, fts))

        self.lins = nn.ModuleList()
        for i in range(n_lin):
            if i == n_lin-1:
                self.lins.append(nn.Linear(fts, last_fts))
            else:
                self.lins.append(nn.Linear(fts, fts))

    def forward(self, x, A):

        sparse_adj = pyg.utils.dense_to_sparse(torch.Tensor(A))
        self.edge_idx = sparse_adj[0]
        self.edge_weights = sparse_adj[1]

        x_out = x
        for i, conv in enumerate(self.convs):
            x_out = conv(x_out, self.edge_idx, self.edge_weights)
            if len(self.lins) != 0 or i < len(self.convs)-1:
                x_out = self.act(x_out)    

        for i, linear in enumerate(self.lins):
            x_out = linear(x_out)
            if i < len(self.lins)-1:
                x_out = self.act(x_out)

        if self.last_act:
            x_out = self.last_act(x_out)
            
        return x_out.squeeze()
    

# 
# class GCNN(nn.Module):
#     def __init__(self, fts, A, input, n_convs=3, n_lin=0, act=nn.ReLU(),
#                  last_act=None, last_fts=1):
#         super(GCNN, self).__init__()
#         N = A.shape[0]
#         if len(input.shape) == 1:
#             self.input = torch.Tensor(input).reshape((N, 1))
#             inp_fts = 1
#         else:
#             self.input = torch.Tensor(input)
#             inp_fts = input.shape[1]

#         self.act = act
#         self.last_act = last_act
#         self.convs = nn.ModuleList()
#         self.convs.append(pyg.nn.GCNConv(inp_fts, fts))
#         for i in range(1, n_convs):
#             if (i == n_convs-1) and n_lin == 0:
#                 self.convs.append(pyg.nn.GCNConv(fts, last_fts))
#             else:
#                 self.convs.append(pyg.nn.GCNConv(fts, fts))

#         self.lins = nn.ModuleList()
#         for i in range(n_lin):
#             if i == n_lin-1:
#                 self.lins.append(nn.Linear(fts, last_fts))
#             else:
#                 self.lins.append(nn.Linear(fts, fts))

#         sparse_adj = pyg.utils.dense_to_sparse(torch.Tensor(A))
#         self.edge_idx = sparse_adj[0]
#         self.edge_weights = sparse_adj[1]

#     def forward(self, x):
#         x_out = x
#         for i, conv in enumerate(self.convs):
#             x_out = conv(x_out, self.edge_idx, self.edge_weights)
#             if len(self.lins) != 0 or i < len(self.convs)-1:
#                 x_out = self.act(x_out)    

#         for i, linear in enumerate(self.lins):
#             x_out = linear(x_out)
#             if i < len(self.lins)-1:
#                 x_out = self.act(x_out)

#         if self.last_act:
#             x_out = self.last_act(x_out)
            
#         self.x_hat = x_out.squeeze().cpu().detach().numpy()
#         return x_out.squeeze()

#     def count_params(self):
#         return sum(p.numel() for p in self.model.parameters()
#                    if p.requires_grad)


# Optimizer constans
SGD = 1
ADAM = 0

class Model:
    def __init__(self, arch,
                 learning_rate=0.001, loss_func=nn.MSELoss(reduction='none'),
                 epochs=1000, opt=ADAM):
        assert opt in [SGD, ADAM], 'Unknown optimizer type'
        self.arch = arch
        self.loss = loss_func
        self.epochs = epochs
        if opt == ADAM:
            self.optim = optim.Adam(self.arch.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.arch.parameters(), lr=learning_rate)

    def fit(self, x_n):


        best_err = 1000000
        best_net = None

        for i in range(1, self.epochs+1):
            self.arch.zero_grad()

            x_hat = self.arch(self.arch.input)

            loss = self.loss(x_hat, x_n)
            loss_red = loss.mean()

            if best_err > 1.005*loss_red:
                best_err = loss_red
                best_net = copy.deepcopy(self.arch)

            loss_red.backward()
            self.optim.step()

        self.arch = best_net

        return x_hat