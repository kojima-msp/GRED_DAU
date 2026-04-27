import numpy as np
import glob
import os
import open3d as o3d
import dgl
import torch
import fpsample

def preprocess_3dpc(filepath,max_points=1000):

    mesh_original = o3d.io.read_triangle_mesh(filepath)

    # make dense 3dpc
    mesh_dense = mesh_original.subdivide_loop(number_of_iterations=2)

    
    data_node = np.asarray(mesh_dense.vertices)
    mask = ~np.isnan(data_node).any(axis=1)
    data_node = data_node[mask]

    # fpsample
    data_node = data_node[fpsample.fps_sampling(data_node, max_points)]

    data_node = min_max(data_node)
    data_node *= 100  # Scale the points

    N = data_node.shape[0] 
    
    # calc knn graph
    K = 5
    knn_g = dgl.knn_graph(torch.tensor(data_node).cpu(),K,exclude_self=True)
    edge_idx_set = set([(min(int(e1), int(e2)), max(int(e1), int(e2))) for e1, e2 in zip(knn_g.edges()[0], knn_g.edges()[1])])

    W = np.zeros((N,N))
    for edge in edge_idx_set:
        W[edge[0],edge[1]] = 1/(np.linalg.norm(data_node[edge[0]]-data_node[edge[1]],ord=2)+1e-10)
    W = W + W.T
    W = W/np.max(W)
    
    D = np.diag(np.ravel(W.sum(1)), 0)
    L = D - W

    return data_node,L,W

def make_data(mode, N_sample=10, max_points=500, noise_list = np.arange(10, 31, 5)):

    os.makedirs(f'datasets/npz_files/modelnet/{mode}', exist_ok=True)

    file_path_list = glob.glob(f'datasets/ModelNet10/*/{mode}/*.off')
    np.random.shuffle(file_path_list)
    file_path_list = file_path_list[:N_sample]
    
    for f in file_path_list:

        data_node,L,A = preprocess_3dpc(f, max_points=max_points)

        for noise_obs in noise_list:

            # add noise
            data_node_obs = data_node + np.random.normal(loc=0, scale=noise_obs, size=data_node.shape)  
            
            np.savez('datasets/npz_files/modelnet/{}/{}_{}'.format(mode, os.path.basename(f).split('.')[0],noise_obs), data_node=data_node, data_node_obs=data_node_obs, L=L, A=A)

def min_max(x):
    v_min = x.min()
    v_max = x.max()
    result = (x-v_min)/(v_max-v_min)
    return result

if(__name__=='__main__'):

    max_points = 500     # number of sampling points
    NOISE_LIST = np.arange(10, 31, 5)


    # make_data('train', N_sample=10, max_points=max_points, noise_list=NOISE_LIST)
    # make_data('test', N_sample=10, max_points=max_points, noise_list=NOISE_LIST)
    
    # for visualization
    f = 'datasets/ModelNet10/chair/test/chair_0969.off'
    os.makedirs(f'datasets/npz_files/modelnet/visualization', exist_ok=True)
    data_node,L,A = preprocess_3dpc(f, max_points=max_points)

    for noise_obs in NOISE_LIST:
        # add noise
        data_node_obs = data_node + np.random.normal(loc=0, scale=noise_obs, size=data_node.shape)          
        np.savez('datasets/npz_files/modelnet/visualization/{}_{}'.format(os.path.basename(f).split('.')[0],noise_obs), data_node=data_node, data_node_obs=data_node_obs, L=L, A=A)


