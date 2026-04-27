import torch
import glob
import sys
import os
import optuna
import click
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch import Generator

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models'))
from proposed import Graph_RED

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from load_data import GraphDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

@click.command()
@click.option('--denoiser', type=click.Choice(['lr', 'pnp']), default='pnp')
@click.option('--dataset', type=click.Choice(['bandlimited', 'modelnet']), default='modelnet')
@click.option('--layers', type=click.INT, default=10, help='Number of layers')
def main(denoiser, dataset, layers):
    # objective関数の外でデータセットを準備
    train_flist = glob.glob(f'datasets/npz_files/{dataset}/train/*.npz')
    train_dataset = GraphDataset(train_flist)
    g = Generator(device=device)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, generator=g) # バッチサイズとnum_workersを設定

    model = Graph_RED(layers=layers,denoiser_name=denoiser)

    @torch.no_grad()
    def objective(trial):
        total_loss = 0

        alpha_red = trial.suggest_float('alpha_red',-4,4)
        alpha_lr = trial.suggest_float('alpha_lr',-4,4)

        if(denoiser=='lr'):
            init_params = {'red':alpha_red,'lr':alpha_lr}
        else:
            alpha_pnp = trial.suggest_float('alpha_pnp',-4,4)
            init_params = {'red':alpha_red,'lr':alpha_lr,'pnp':alpha_pnp}
        
        model.set_params(init_params)

        for data_node, data_node_obs, L, A in train_loader:
            data_node = data_node.squeeze(0)
            data_node_obs = data_node_obs.squeeze(0)
            L = L.squeeze(0)

            x_tilde = model.forward(data_node_obs, L)
            total_loss += mean_squared_error(data_node.cpu().detach().numpy(), x_tilde.cpu().detach().numpy())

        return total_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    best_params = study.best_trial.params

    save_dir = 'src/exp/train/pretrained_weights'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'proposed_{denoiser}_{dataset}_{layers}.npz'
    save_path = os.path.join(save_dir, file_name)
    np.savez(save_path, **best_params)

if(__name__=='__main__'):
    main()
