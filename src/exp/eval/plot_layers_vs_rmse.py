import torch
import glob
import os
import numpy as np
from sklearn.metrics import root_mean_squared_error
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models'))
from proposed import Graph_RED
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee','grid','high-vis'])

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from load_data import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

def exe():

    LAYER_LIST = [5,10,15,20,25]
    dataset = 'bandlimited'

    models_dict = dict()

    indices = ['RED (LR)', 'RED (LR-DAU)', 'RED (PnP)', 'RED (PnP-DAU)']
    markers = ['o', '<', '*', '>', 's']
    df = pd.DataFrame(columns=LAYER_LIST, index=indices)
    df = df.fillna(0)

    for layer in LAYER_LIST:
        
        models_dict[layer] = dict()
        
        for denoiser in ['lr', 'pnp']:
            
            # model-based
            params = dict()
            param = np.load(f'src/exp/train/pretrained_weights/proposed_{denoiser}_{dataset}_{layer}.npz')
            if denoiser=='lr':
                params['red'] = param['alpha_red']
                params['lr'] = param['alpha_lr']

            else:
                params['red'] = param['alpha_red']
                params['pnp'] = param['alpha_pnp']
                params['lr'] = param['alpha_lr']


            proposed = Graph_RED(layers=layer, denoiser_name=denoiser, init_params=params).eval()
            models_dict[layer][f'RED ({denoiser.replace("lr", "LR").replace("pnp", "PnP")})'] = proposed

            # dau
            proposed_dau = Graph_RED(layers=layer, denoiser_name=denoiser)
            proposed_dau.eval()
            proposed_dau.load_state_dict(torch.load(f'src/exp/train/pretrained_weights/proposed_{denoiser}_{dataset}_dau_{layer}.pth'))
            models_dict[layer][f'RED ({denoiser.replace("lr", "LR").replace("pnp", "PnP")}-DAU)'] = proposed_dau

    test_flist = glob.glob(f"datasets/npz_files/{dataset}/test/*.npz")

    for f in test_flist:
                
        data_node_gt, data_node_obs, L, A = load_data(f)
        data_node_obs = torch.tensor(data_node_obs)
        L = torch.tensor(L)

        for layer in LAYER_LIST:

            # df.loc['Observed', layer] += root_mean_squared_error(data_node_gt, data_node_obs.cpu().numpy())

            models = models_dict[layer]
            for k, model in models.items(): 
                x_tilde = model.forward(data_node_obs,L)
                df.loc[k, layer] += root_mean_squared_error(data_node_gt, x_tilde.detach().cpu().numpy().squeeze())
    
    df = df / len(test_flist)
    df.to_csv('out/ablation_layers_vs_rmse.csv')
    
    data = df.values

    for methodname, score, marker in zip(indices, data, markers):
        plt.plot(LAYER_LIST, score, label=methodname, marker=marker)

    plt.xlabel(r'$K$')
    plt.ylabel('RMSE')
    plt.xticks(LAYER_LIST)
    plt.legend()
    plt.savefig('out/img/ablation_layers_vs_rmse.pdf')



if(__name__ == "__main__"):
    exe()