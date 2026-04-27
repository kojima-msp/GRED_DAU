import glob
import os
import sys

import click
import numpy as np
from pygsp.graphs import Graph
import torch
from torch import nn

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee','grid','high-vis'])

plt.rcdefaults()
plt.rcParams["font.size"] = 18
plt.rcParams['axes.axisbelow'] = True

sys.path.append(".")
from src.models.PnP import Graph_PnP
from src.utils.load_data import load_data

sys.path.append("src/models/Graph_Deep_Decoder")
from graph_deep_decoder import utils
from graph_deep_decoder.model import Model, select_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

@click.command()
@click.option('--dataset', type=click.Choice(['bandlimited', 'modelnet']), prompt='Type dataset name', help='Dataset name')
def main(dataset):

    params_pnp = np.load(f'src/exp/train/pretrained_weights/pnp_{dataset}.npz')
    params = dict()
    params['pnp'] = params_pnp['alpha_pnp']
    params['lr'] = params_pnp['alpha_lr']

    results_pnp = list()
    results_gat = list()

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    c = 1.1

    score_pnp = list()
    score_gat = list()
    score_pnp_c = list()
    score_gat_c = list()

    flist = glob.glob(f"datasets/npz_files/{dataset}/train/*.npz")
    flist.sort()

    # for fname in flist:
    for fname in flist:

        _, data_node, L, A = load_data(fname)
        G = Graph(A)
        
        for d in range(data_node.shape[1]):
            
            data_node_obs = data_node[:,d][:,np.newaxis]

            # pnp
            x_tilde = Graph_PnP(layers=10).forward(torch.tensor(data_node_obs),torch.tensor(L),alpha_pnp=params['pnp'],alpha_lr=params['lr']).cpu().detach().numpy()
            x_tilde_c = Graph_PnP(layers=10).forward(torch.tensor(data_node_obs*c),torch.tensor(L),alpha_pnp=params['pnp'],alpha_lr=params['lr']).cpu().detach().numpy()
            score_pnp.extend(c*x_tilde)
            score_pnp_c.extend(x_tilde_c)

            # exsitng
            exp = {'type': 'GAT', 'fts': 50, 'A': A, 'heads': 3, 'last_fts': 1, 'last_act': None,
            'loss': nn.MSELoss(reduction='none'), 'legend': 'GAT'}
            model = select_model(exp, data_node_obs, 500, 0.1)
            model.fit(data_node_obs)
            x_tilde_gat = model.test(torch.tensor(data_node_obs).cpu())
            x_tilde_gat_c = model.test(torch.tensor(c * data_node_obs).cpu())
            score_gat.extend(c*x_tilde_gat)
            score_gat_c.extend(x_tilde_gat_c)

    ax1.scatter(score_gat, score_gat_c, label='GAT', marker='o', color='purple')
    ax2.scatter(score_pnp, score_pnp_c, label='PnP', marker='o', color='orange')
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_xticklabels([0, 25, 50, 75, 100])
    ax1.set_yticklabels([0, 25, 50, 75, 100])
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels([0, 25, 50, 75, 100])
    ax2.set_yticklabels([0, 25, 50, 75, 100])
    
    ax1.set_xlabel(r'$c\mathcal{D}_\text{GAT}(\mathbf{x})$')
    ax1.set_ylabel(r'$\mathcal{D}_\text{GAT}(c\mathbf{x})$')
    ax2.set_xlabel(r'$c\mathcal{D}_\text{PnP}(\mathbf{x})$')
    ax2.set_ylabel(r'$\mathcal{D}_\text{PnP}(c\mathbf{x})$')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'out/img/condition1_{dataset}.pdf')

if __name__ == "__main__":
    main()
