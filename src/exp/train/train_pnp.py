import torch
from torch import nn
import numpy as np
import glob
import tqdm
import sys
import os
import optuna
import click
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch import Generator

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models'))
from PnP import Graph_PnP

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from load_data import GraphDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

@click.command()
@click.option('--dataset', type=click.Choice(['bandlimited', 'modelnet']), default='modelnet')
def main(dataset):
    # objective関数の外でデータセットとモデルを準備
    train_flist = glob.glob(f'datasets/npz_files/{dataset}/train/*.npz')
    train_dataset = GraphDataset(train_flist)
    g = Generator(device=device)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, generator=g)

    model = Graph_PnP(layers=10)

    def objective(trial):
        loss = 0
        alpha_lr = trial.suggest_float('alpha_lr', -4, 4)
        alpha_pnp = trial.suggest_float('alpha_pnp', -4, 4)

        for data_node, data_node_obs, L, A in train_loader:
            data_node = data_node.squeeze(0)
            data_node_obs = data_node_obs.squeeze(0)
            L = L.squeeze(0)

            model.reset_params()
            x_tilde = model.forward(data_node_obs, L, torch.tensor(alpha_pnp), torch.tensor(alpha_lr))
            loss += mean_squared_error(data_node.cpu().detach().numpy(), x_tilde.cpu().detach().numpy())
            
        return loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    best_params = study.best_trial.params

    save_dir = 'src/exp/train/pretrained_weights'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'pnp_{dataset}.npz'
    save_path = os.path.join(save_dir, file_name)
    np.savez(save_path, **best_params)

if(__name__=='__main__'):
    main()
