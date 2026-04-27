import torch
import numpy as np
import glob
import tqdm
import click
import sys
from random import shuffle
import os
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
@click.option('--dataset', type=click.Choice(['bandlimited', 'modelnet']), prompt='Type dataset name', help='Dataset name')
@click.option('--denoiser', type=click.Choice(['lr', 'pnp']), prompt='Type denoiser name', help='Denoiser name')
@click.option('--layers', type=click.INT, default=10, help='Number of layers')
def main(dataset, denoiser, layers):
    # experimental setting
    lr = 1e-3
    EPOCHS = 500

    min_mse = 10**6

    model = Graph_RED(layers=layers, denoiser_name=denoiser)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.MSELoss()

    train_flist = glob.glob(f'datasets/npz_files/{dataset}/train/*.npz')
    train_dataset = GraphDataset(train_flist)
    g = Generator(device=device)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, generator=g)



    range_epoch = range(EPOCHS)
    for epoch in tqdm.tqdm(range_epoch):
        loss_sum = torch.zeros(1)

        for data_node, data_node_obs, L, _ in train_loader:

            data_node = torch.tensor(data_node.squeeze(0)[:,np.newaxis])
            data_node_obs = torch.tensor(data_node_obs.squeeze(0)[:,np.newaxis])
            L = torch.tensor(L.squeeze(0))

            x_tilde = model.forward(data_node_obs,L)
            loss = criterion(x_tilde,data_node)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach()

        if(loss_sum < min_mse):
            min_mse = loss_sum
            model_best = model.state_dict()

            tqdm.tqdm.write(f'{epoch}: {loss_sum} (best)')
            model_best = model.state_dict()

        else:
            tqdm.tqdm.write(f'{epoch}: {loss_sum}')
            
    torch.save(model_best,f'src/exp/train/pretrained_weights/proposed_{denoiser}_{dataset}_dau_{layers}.pth')

if(__name__=='__main__'):
    main()