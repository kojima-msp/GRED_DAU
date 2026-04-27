from torch import optim, no_grad, nn, Tensor
import networkx as nx
import copy
import time
import numpy as np

import torch

import cvxpy as cp

from graph_deep_decoder import utils
from graph_deep_decoder.architecture import (GFUps, GraphDecoder,
                                             GraphDeepDecoder)
from graph_deep_decoder.baselines import GCNN, GAT, KronAE, GUTF

# For unrolling
# import GUN.models

# Optimizer constans
SGD = 1
ADAM = 0


class Model:
    def __init__(self, arch,
                 learning_rate=0.001, loss_func=nn.MSELoss(reduction='none'),
                 epochs=1000, eval_freq=100, verbose=False,
                 opt=ADAM):
        assert opt in [SGD, ADAM], 'Unknown optimizer type'
        self.arch = arch
        self.loss = loss_func
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.verbose = verbose
        if opt == ADAM:
            self.optim = optim.Adam(self.arch.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.arch.parameters(), lr=learning_rate)

    def count_params(self):
        ps = sum(p.numel() for p in self.arch.parameters() if p.requires_grad)
        return ps

    def get_filter_coefs(self):
        filter_coefs = []
        for layer in self.arch.model:
            if isinstance(layer, GFUps):
                filter_coefs.append(layer.hs.detach().numpy())
        return np.array(filter_coefs)

    def classif_err(self, x_hat, x):
        if len(x.shape) > 1:
            x_hat_aux = torch.argmax(x_hat, dim=1)
            x_aux = torch.argmax(x, dim=1)
            eval_loss = torch.abs(x_hat_aux - x_aux)/x.shape[0]
            return eval_loss.detach().cpu().numpy()
        else:
            x_hat_label = torch.round(x_hat)
            # Set maximum allowed label
            max_label = x.max()
            x_hat_label[x_hat_label > max_label] = max_label
            eval_loss = torch.ne(x_hat_label, x)/x.shape[0]
            return eval_loss.detach().cpu().numpy()

    def fit(self, signal, x=None, reduce_err=True, class_val=False, adj_list=None):
        # NOTE: true signal x will only be used to obtain the true error for
        # each iteration to plot it, but it must not influence the learning of
        # the parameters
        if x is not None:
            x = Tensor(x)
            if adj_list is not None:
                x = x.reshape((x.size(0), 1))

        x_n = Tensor(signal)

        best_err = 1000000
        best_net = None
        best_epoch = 0
        train_err = np.zeros((self.epochs, signal.shape[0]))
        val_err = np.zeros((self.epochs, signal.shape[0]))
        for i in range(1, self.epochs+1):
            t_start = time.time()
            self.arch.zero_grad()

            if adj_list is None:
                # Our models
                self.arch.cpu()
                x_hat = self.arch(self.arch.input)
            else:
                # IN UNROLLING
                x_hat = self.arch(x_n, adj_list)
                if x_hat.dim() == 3:
                    x_hat = x_hat[:, 0, :]

            self.x_hat = x_hat.cpu().detach().numpy()

            loss = self.loss(x_hat, x_n)
            loss_red = loss.mean()

            if best_err > 1.005*loss_red:
                best_epoch = i
                best_err = loss_red
                best_net = copy.deepcopy(self.arch)

            loss_red.backward()
            self.optim.step()

        self.arch = best_net

        if reduce_err:
            train_err = np.sum(train_err, axis=1)
            val_err = np.sum(val_err, axis=1)

        return train_err, val_err, best_epoch

    def test(self, x, classif=False):
        x_hat = self.arch(x).squeeze()

        return x_hat.cpu().detach().numpy()


class CVXModel:
    def __init__(self):
        self.x_hat = None

    def count_params(self):
        return 0

    def fit(self, signal):
        raise Exception('Method "fit" not implemented.')

    def test(self, x):
        err = np.linalg.norm(x-self.x_hat)**2/np.linalg.norm(x)**2
        node_err = err/x.size
        return node_err, err

    def test_classification(self, x):
        x_hat_label = np.round(self.x_hat)
        max_label = x.max()
        x_hat_label[x_hat_label > max_label] = max_label
        node_err = np.not_equal(x_hat_label, x)/x.shape[0]
        return node_err, np.sum(node_err)


class BLModel(CVXModel):
    def __init__(self, V, coefs):
        super(BLModel, self).__init__()
        self.V = V
        self.k = min(coefs, V.shape[0])

    def count_params(self):
        return self.coefs

    def fit(self, signal):
        self.x_hat = self.V[:, :self.k].dot(self.V[:, :self.k].T.dot(signal))


class MeanModel(CVXModel):
    def __init__(self, A):
        super(MeanModel, self).__init__()
        self.A = A
        self.D_inv = np.diag(1/self.A.sum(axis=1))

    def fit(self, signal):
        self.x_hat = self.D_inv.dot(self.A).dot(signal)


# Total Variation/Graph Filtering model
class TVModel(CVXModel):
    def __init__(self, A, alhpa):
        super(TVModel, self).__init__()
        assert np.isreal(A).all(), 'Only real matrices supported'
        eig_vals, _ = np.linalg.eig(A)
        self.A = A/np.max(np.abs(eig_vals))
        self.alpha = alhpa

    def fit(self, signal):
        assert np.isreal(signal).all(), 'Only real signals supported'
        Eye = np.eye(signal.size)
        A_hat = Eye - self.A
        H_A = np.linalg.inv(Eye+self.alpha*A_hat.T.dot(A_hat))
        self.x_hat = np.asarray(H_A.dot(signal))


# Laplazian Regularizer
class LRModel(CVXModel):
    def __init__(self, L, alhpa):
        super(LRModel, self).__init__()
        self.L = L
        self.alpha = alhpa

    def fit(self, signal):
        assert np.isreal(signal).all(), 'Only real signals supported'
        Eye = np.eye(signal.size)
        self.x_hat = np.linalg.inv(Eye+self.alpha*self.L).dot(signal)
        self.x_hat = np.asarray(self.x_hat)


# median filter
class MedianModel(CVXModel):
    def __init__(self, A):
        super(MedianModel, self).__init__()
        self.A = A + np.eye(A.shape[0])

    def fit(self, x):
        self.x_hat = np.zeros(x.shape)
        for i in range(self.A.shape[0]):
            neighbours_ind = np.asarray(self.A[i, :] != 0).nonzero()
            neighbours = x[neighbours_ind]
            self.x_hat[i] = np.median(neighbours)


# Trend Filtering model
class GTFModel(CVXModel):
    def __init__(self, A, k, lamb):
        super(GTFModel, self).__init__()
        L = np.diag(np.sum(A, axis=0)) - A
        self.lamb = lamb
        if k % 2 == 0:
            G = nx.from_numpy_array(A)
            M = nx.linalg.graphmatrix.incidence_matrix(G).todense()
            self.Delta = M.T@np.linalg.matrix_power(L, int(k/2))
        else:
            self.Delta = np.linalg.matrix_power(L, int((k-1)/2))

    def fit(self, x):
        self.x_hat = x
        x_hat = cp.Variable(x.size)
        obj = cp.Minimize(.5*cp.sum_squares(x-x_hat) +
                          self.lamb*cp.norm(self.Delta@x_hat, 1))

        prob = cp.Problem(obj)
        try:
            prob.solve()
        except cp.SolverError:
            print('WARNING: solver error')
            return

        if prob.status not in ['optimal', 'optimal_inaccurate']:
            print('WARNING:', prob.status)
            return

        self.x_hat = x_hat.value


def select_model(exp, x_n, epochs, lr):
    if exp['type'] == 'TV':
        return TVModel(exp['A'], exp['alpha'])

    elif exp['type'] == 'LR':
        return LRModel(exp['L'], exp['alpha'])

    elif exp['type'] == 'BL':
        _, V = utils.ordered_eig(exp['S'])
        N = V.shape[0]
        coefs = int(N*exp['alpha']) if exp['alpha'] <= 1 else exp['alpha']
        return BLModel(V, coefs)

    elif exp['type'] == 'MED':
        return MedianModel(exp['S'])

    elif exp['type'] == 'GTF':
        return GTFModel(exp['A'], exp['k'], exp['lamb'])
    else:
        if exp['type'] == '2LD':
            dec = GraphDecoder(exp['fts'], exp['H'], exp['std'])

        elif exp['type'] == 'DD':
            gamma = exp['gamma'] if 'gamma' in exp.keys() else .5
            dec = GraphDeepDecoder(exp['fts'], exp['nodes'], exp['Us'],
                                   batch_norm=exp['bn'], As=exp['As'],
                                   act_fn=exp['af'], ups=exp['ups'],
                                   last_act_fn=exp['laf'],
                                   input_std=exp['in_std'], w_std=exp['w_std'],
                                   gamma=gamma)

        elif exp['type'] == 'GCNN':
            n_convs = exp['n_convs'] if 'n_convs' in exp.keys() else 3
            n_lin = exp['n_lin'] if 'n_lin' in exp.keys() else 0
            act = exp['act'] if 'act' in exp.keys() else nn.ReLU()
            dec = GCNN(exp['fts'], exp['A'], x_n, last_fts=exp['last_fts'],
                       last_act=exp['last_act'], act=act,
                       n_convs=n_convs, n_lin=n_lin)

        elif exp['type'] == 'GAT':
            n_at = exp['n_at'] if 'n_at' in exp.keys() else 1
            n_lin = exp['n_lin'] if 'n_lin' in exp.keys() else 2
            act = exp['act'] if 'act' in exp.keys() else nn.ReLU()
            dec = GAT(exp['fts'], exp['A'], exp['heads'], x_n, 
                      n_at=n_at, n_lin=n_lin, act=act)

        elif exp['type'] == 'KronAE':
            dec = KronAE(exp['fts'], exp['A'], exp['r'], x_n)

            # arch = arch.cuda()
        else:
            raise Exception('Unkwown exp type')

        if 'loss' in exp.keys():
            return Model(dec, epochs=epochs, learning_rate=lr,
                         loss_func=exp['loss'])

        return Model(dec, epochs=epochs, learning_rate=lr)