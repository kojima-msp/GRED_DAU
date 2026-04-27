import numpy as np
import pygsp
import matplotlib.pyplot as plt
import os
from sklearn.metrics import root_mean_squared_error
import click

import sys
sys.path.append("src/utils")
from plotting import plot_bandlimited, plot_3dpc

@click.command()
@click.option('--dataset', type=click.Choice(['bandlimited', 'modelnet']), prompt='Type dataset name', help='Dataset name')
@click.option('--noise_level', type=click.Choice(['10','15','20','25','30']), prompt='Type noise level', help='Noise level', default='10')
@click.option('--mode', type=click.Choice(['original', 'diff']), prompt='Type mode', help='Mode', default='diff')
def main(dataset,noise_level,mode):

    noise_level = int(noise_level)

    # load dataset
    # if dataset=='bandlimited', target is 000~004
    # if dataset=='modelnet', target is chair_09xx (xx: 14, 19, 34, 81, 85)
    path_target = f'datasets/npz_files/bandlimited/test/001_{noise_level}.npz' if dataset == 'bandlimited' else f'datasets/npz_files/modelnet/visualization/chair_0969_{noise_level}.npz'

    dataset = path_target.split('/')[-3]
    os.makedirs(f'out/img/visualization/{dataset}', exist_ok=True)

    # load groundtruth and coordinates
    data_original = np.load(path_target)
    A = data_original['A']
    coords = data_original['coords'] if dataset == 'bandlimited' else data_original['data_node']


    data_node = data_original['data_node'][:,np.newaxis] if dataset == 'bandlimited' else data_original['data_node']

    # set fuction to plot
    plot_func = plot_bandlimited if dataset == 'bandlimited' else plot_3dpc

    # plot original data
    plot_func(A, coords, data_node, f'out/img/visualization/{dataset}/{mode}_original_{noise_level}.pdf')

    # load denoised data
    path_denoised = path_target.replace('datasets/npz_files','out/npz').replace('test/','').replace('visualization/','')
    data_denoised = np.load(path_denoised)

    # search max and min values
    results = [np.abs(data_node-signal) for signal in data_denoised.values()] if mode == 'diff' else [signal for signal in data_denoised.values()]
    vmax = noise_level
    vmin = np.min(results)

    outstr = ''

    for legend,signal in data_denoised.items():
        legend = legend.replace('(','_').replace(')','_').replace(',','_').replace(' ','')
        if mode == 'diff':
            coords = coords if dataset=='bandlimited' else signal
            plot_func(A, coords, np.abs(data_node-signal), f'out/img/visualization/{dataset}/{mode}_{legend}_{noise_level}.pdf', vmax=vmax, vmin=vmin)
        else:
            plot_func(A, coords, signal, f'out/img/visualization/{dataset}/{mode}_{legend}_{noise_level}.pdf', vmax=vmax, vmin=vmin)
        
        print(f'{legend} RMSE:{root_mean_squared_error(data_node,signal)}')
        outstr += f'{legend}: {root_mean_squared_error(data_node,signal)}\n'

    with open(f'out/img/visualization/{dataset}/rmse_{noise_level}.txt','w') as f:
        f.write(outstr)

if __name__ == "__main__":
    main()