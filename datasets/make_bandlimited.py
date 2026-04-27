import numpy as np
from sklearn.neighbors import NearestNeighbors
import pygsp
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def make_bandlimited_graph(N, K = 5, coord_max = 100, coord_min = 0, N_bandlimit = 10, omega = 0, DC = 0):

    rng = np.random.default_rng()

    # gen 100x100 meshgrid
    X, Y = np.meshgrid(np.linspace(coord_min, coord_max-1, coord_max-coord_min), np.linspace(coord_min, coord_max-1, coord_max-coord_min))
    coord_set = np.array([X.flatten(), Y.flatten()],dtype=int).T

    # randomly select N nodes
    coords = [tuple(node) for node in rng.choice(coord_set, N,replace=False).tolist()]

    # add edges using k-nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=K).fit(coords)
    A = nbrs.kneighbors_graph(coords,mode='distance').toarray()

    A = np.where(A>0,1/A,0)
    
    # gen graph
    G = pygsp.graphs.Graph(A)
    G.set_coordinates(coords)

    # calc laplacian and eigenvalues
    G.compute_fourier_basis()

    # calc sin matrix
    dic = np.ones(N_bandlimit)*omega*np.pi
    for i in range(N_bandlimit):
        phi = i/N_bandlimit * np.pi
        dic[i] = dic[i]+phi
    dic = np.sin(dic) + DC
    
    # gen bandlimited signal
    signal = G.U[:,:N_bandlimit] @ dic

    # scale signal to [0,100]
    signal = (signal - np.min(signal))/(np.max(signal) - np.min(signal)) * 100
    
    return G, signal

def make_npz(N, path):
    NOISE_LIST = np.arange(10,31,5)
    N_nodes = 100

    os.makedirs(f'datasets/npz_files/bandlimited/{path}', exist_ok=True)
    
    for k in range(N):
        G, data_node = make_bandlimited_graph(N_nodes, N_bandlimit=3, DC=2)

        for noise_obs in NOISE_LIST:
            data_node_obs = data_node + np.random.normal(loc=0, scale=noise_obs, size=data_node.shape)

            np.savez('datasets/npz_files/bandlimited/{}/{}_{}'.format(path,str(k).zfill(3),noise_obs), data_node=data_node, data_node_obs=data_node_obs, L=G.L.toarray(), A=np.array(G.W.toarray()), coords=G.coords)


if(__name__ == '__main__'):
    make_npz(10, 'train')
    make_npz(5, 'test')

    # debug
    # plt.set_cmap('jet')
    # G, signal = make_bandlimited_graph(100,N_bandlimit=3, DC=2)
    # G.plot_signal(signal)
    # plt.savefig('bandlimited.png')