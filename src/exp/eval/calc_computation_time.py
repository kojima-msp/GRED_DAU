import torch
import glob
import os
import numpy as np
from sklearn.metrics import root_mean_squared_error
from torch import nn
from pygsp.graphs import Graph
import sys
import click
import time

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
torch.set_default_device(device)

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

        result = dict()
        result['LR'] = list()
        result['PnP'] = list()
        result['Proposed (LR)'] = list()
        result['Proposed (PnP)'] = list()
        result['Proposed (LR, DAU)'] = list()
        result['Proposed (PnP, DAU)'] = list()

        for f in test_flist:
            
            _, data_node_obs, L, A = load_data(f)

            data_node_obs = torch.tensor(data_node_obs)
            L = torch.tensor(L)

            # LR
            ts= time.time()
            LaplacianRegularization().forward(data_node_obs,L,alpha=torch.tensor(params['lr']))
            result['LR'].append(time.time() - ts)

            # pnp
            ts= time.time()
            Graph_PnP(layers=10).forward(data_node_obs,L,alpha_pnp=torch.tensor(params['pnp']['pnp']),alpha_lr=torch.tensor(params['pnp']['lr']))
            result['PnP'].append(time.time() - ts)

            # Propsoed (LR)
            ts= time.time()
            proposed_lr.forward(data_node_obs,L)
            result['Proposed (LR)'].append(time.time() - ts)

            # Propsoed (pnp)
            ts= time.time()
            proposed_pnp.forward(data_node_obs,L)
            result['Proposed (PnP)'].append(time.time() - ts)

            # Propsoed (DAU, LR)
            ts= time.time()
            proposed_lr_dau.forward(data_node_obs,L)
            result['Proposed (LR, DAU)'].append(time.time() - ts)

            # Propsoed (DAU, pnp)
            ts= time.time()
            proposed_pnp_dau.forward(data_node_obs,L)
            result['Proposed (PnP, DAU)'].append(time.time() - ts)
        
    for k,v in result.items():
            if dataset=='modelnet':
                print(k, np.array(v).mean()/3) # 3 channels
            else:
                print(k, np.array(v).mean())

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