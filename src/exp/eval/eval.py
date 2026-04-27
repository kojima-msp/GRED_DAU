import torch
import glob
import os
import numpy as np
from sklearn.metrics import root_mean_squared_error
from torch import nn
from pygsp.graphs import Graph
import sys
import click

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models'))
from existing import LaplacianRegularization
from PnP import Graph_PnP
from proposed import N2N_DAU_RED, Graph_RED

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from load_data import load_data

sys.path.append(os.path.join(os.path.dirname(__file__), "../../models/Graph_Deep_Decoder"))
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import Ups

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
# torch.set_default_device(device)
torch.set_default_device('cpu')

def exe(dataset,params):

    NOISE_LIST = np.arange(10,31,5)

    # proposed baseline
    proposed_lr = Graph_RED(layers=10, denoiser_name='lr', init_params=params['proposed_lr']).eval()
    proposed_pnp = Graph_RED(layers=10, denoiser_name='pnp', init_params=params['proposed_pnp']).eval()

    proposed_lr_dau = Graph_RED(layers=10, denoiser_name='lr')
    proposed_lr_dau.eval()
    proposed_lr_dau.load_state_dict(torch.load(params['proposed_lr_dau']))

    proposed_pnp_dau = Graph_RED(layers=10, denoiser_name='pnp')
    proposed_pnp_dau.eval()
    proposed_pnp_dau.load_state_dict(torch.load(params['proposed_pnp_dau']))

    for noiselevel in NOISE_LIST:
        test_flist = glob.glob(f"datasets/npz_files/{dataset}/test/*_{noiselevel}.npz")

        # for visualization
        # if dataset=='modelnet':
        #     test_flist += glob.glob(f"datasets/npz_files/{dataset}/visualization/*_{noiselevel}.npz")


        for f in test_flist:
            
            data_node_gt, data_node_obs, L, A = load_data(f)

            data_node_obs = torch.tensor(data_node_obs)
            L = torch.tensor(L)

            G = Graph(A)

            result = dict()

            # observed
            result['Observed'] = data_node_obs.cpu().numpy()

            # LR
            x_tilde = LaplacianRegularization().forward(data_node_obs,L,alpha=torch.tensor(params['lr']))
            result['LR'] = x_tilde.cpu().detach().numpy()

            # pnp
            x_tilde = Graph_PnP(layers=10).forward(data_node_obs,L,alpha_pnp=torch.tensor(params['pnp']['pnp']),alpha_lr=torch.tensor(params['pnp']['lr']))
            result['PnP'] = x_tilde.cpu().detach().numpy()

            # Propsoed (LR)
            x_tilde = proposed_lr.forward(data_node_obs,L)
            result['Proposed (LR)'] = x_tilde.cpu().detach().numpy()

            # Propsoed (pnp)
            x_tilde = proposed_pnp.forward(data_node_obs,L)
            result['Proposed (PnP)'] = x_tilde.cpu().detach().numpy()

            # Propsoed (DAU, LR)
            x_tilde = proposed_lr_dau.forward(data_node_obs,L)
            result['Proposed (LR, DAU)'] = x_tilde.cpu().detach().numpy()

            # Propsoed (DAU, pnp)
            x_tilde = proposed_pnp_dau.forward(data_node_obs,L)
            result['Proposed (PnP, DAU)'] = x_tilde.cpu().detach().numpy()

            # Propsoed (Unsupervised, LR)
            model_dau = N2N_DAU_RED(layers=10, denoiser_name='lr')
            if dataset != 'bandlimited':
                model_dau.model.load_state_dict(torch.load('src/exp/train/pretrained_weights/proposed_lr_bandlimited_dau_10.pth'))
            x_tilde = model_dau.forward(data_node_obs,L,unsupervised_epochs=100)
            result['Proposed (LR, Unsupervised)'] = x_tilde.cpu().detach().numpy()

            # exsitng
            X_hat = utils.run_model({'type': 'GAT', 'fts': 50, 'A': A, 'heads': 3, 'last_fts': 1, 'last_act': None,
            'loss': nn.MSELoss(reduction='none'), 'legend': 'GAT'}, G, data_node_obs.cpu().numpy(), 500, 0.1)
            result['GAT'] = X_hat

            X_hat = utils.run_model({'type': 'DD', 'ups': Ups.U_MEAN, 'nodes': [50, 100, 100] + [G.N]*2, 'fts': [50]*4 + [1],
            'af': nn.ReLU(), 'laf':  None, 'w_std': 1, 'in_std': .5, 'bn': False, 'gamma': .5,
            'loss': nn.MSELoss(reduction='none'), 'legend': 'GD'}, G, data_node_obs.cpu().numpy(), 500, 0.1)
            result['GD'] = X_hat

            for k,v in result.items():
                print(k, root_mean_squared_error(data_node_gt,v))
            
            # save result
            outname = os.path.basename(f).split('.')[0]
            np.savez(f'out/npz/{dataset}/{outname}.npz', **result)

@click.command()
@click.option('--dataset', type=click.Choice(['bandlimited', 'modelnet']), prompt='Type dataset name', help='Dataset name')
def main(dataset):
    os.makedirs(f'out/npz/{dataset}', exist_ok=True)

    params = dict()

    params_lr = np.load(f'src/exp/train/pretrained_weights/lr_{dataset}_10.npz')
    params['lr'] = params_lr['alpha_lr']

    params_pnp = np.load(f'src/exp/train/pretrained_weights/pnp_{dataset}_10.npz')
    params['pnp'] = dict()
    params['pnp']['pnp'] = params_pnp['alpha_pnp']
    params['pnp']['lr'] = params_pnp['alpha_lr']

    params_proposed_lr = np.load(f'src/exp/train/pretrained_weights/proposed_lr_{dataset}_10.npz')
    params['proposed_lr'] = dict()
    params['proposed_lr']['red'] = params_proposed_lr['alpha_red']
    params['proposed_lr']['lr'] = params_proposed_lr['alpha_lr']

    params_proposed_pnp = np.load(f'src/exp/train/pretrained_weights/proposed_pnp_{dataset}_10.npz')
    params['proposed_pnp'] = dict()
    params['proposed_pnp']['red'] = params_proposed_pnp['alpha_red']
    params['proposed_pnp']['pnp'] = params_proposed_pnp['alpha_pnp']
    params['proposed_pnp']['lr'] = params_proposed_pnp['alpha_lr']

    params['proposed_pnp_dau'] = f'src/exp/train/pretrained_weights/proposed_pnp_{dataset}_dau_10.pth'
    params['proposed_lr_dau'] = f'src/exp/train/pretrained_weights/proposed_lr_{dataset}_dau_10.pth'

    exe(dataset,params)

if(__name__ == "__main__"):
    main()