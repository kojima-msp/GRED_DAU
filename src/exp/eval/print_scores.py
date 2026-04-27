import glob
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import os
import click

def round(x, decimals=0):
    return np.floor(x * 10**decimals + 0.5) / 10**decimals

@click.command()
@click.option('--dataset', type=click.Choice(['bandlimited', 'modelnet']), prompt='Type dataset name', help='Dataset name')
def main(dataset):
    pd.options.display.float_format = '{:.2f}'.format

    variance_data = 'data_node'
    variance_range = np.arange(10,31,5)
    METHODS = ['Observed', 'LR', 'PnP', 'Proposed (LR)', 'Proposed (PnP)', 'Proposed (LR, DAU)', 'Proposed (PnP, DAU)', 'GAT', 'GD', 'Proposed (LR, Unsupervised)']
    MTYPE1 = ['']+['Supervised Methods']*6 + ['Unsupervised Methods']*3
    MTYPE2 = ['']+['Model-based']*4 + ['DAU-based']*2 + ['Data-driven']*2 + ['DAU-based']

    # df = pd.DataFrame(index=METHODS, columns=variance_range)
    df = pd.DataFrame(index=[MTYPE1, MTYPE2, METHODS], columns=variance_range)
    df = df.astype('float64')
    df.fillna(0.0, inplace=True)

    for noise_level in variance_range:
        f_orig = glob.glob(f"datasets/npz_files/{dataset}/test/*_{noise_level}.npz")

        for f in f_orig:
            data_orig = np.load(f)
            data_node = data_orig[variance_data]

            f_denoised = f'out/npz/{dataset}/' + os.path.basename(f)

            data_result = np.load(f_denoised)

            for k,method in enumerate(METHODS):
                df.loc[(MTYPE1[k],MTYPE2[k],method), noise_level] += root_mean_squared_error(data_node,data_result[method])
                # df.loc[(method), (noise_level)] += root_mean_squared_error(data_node,data_result[method])

    N_dataset = len(glob.glob(f"datasets/npz_files/{dataset}/test/*.npz"))/len(variance_range)
    print(round(df/N_dataset, 2))
    print(round(df/N_dataset, 2).to_latex().replace('0000','').replace('\\toprule','\\hline').replace('\\midrule','\\hline').replace('\\bottomrule','\\hline'))

if __name__ == "__main__":
    main()