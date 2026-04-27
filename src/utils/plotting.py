import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pygsp

def plot_bandlimited(A, coords, signal, fname,vmax=None, vmin=None):
    plt.set_cmap('turbo')

    if vmax is None:
        vmax = signal.max()
    if vmin is None:
        vmin = signal.min()

    G = pygsp.graphs.Graph(A)
    G.set_coordinates(coords)

    fig = plt.figure()
    ax = fig.add_subplot()
    G.plot_signal(signal,ax=ax,limits=[vmin,vmax])

    ax.set_title('')
    ax.set_xticks(range(0,101,20))
    ax.set_yticks(range(0,101,20))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-5,105)
    ax.set_ylim(-5,105)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_aspect('equal', adjustable='box')
    plt.grid(color = 'gray', linestyle = '-', linewidth = 0.5)

    fig.tight_layout()
    plt.savefig(fname)

def plot_3dpc(A,data,signal,fname, vmax=None, vmin=None):
    plt.set_cmap('turbo')

    fig = plt.figure(figsize=(3,3))

    ax = fig.add_subplot(1,1,1, projection='3d')

    sc = ax.scatter(data[:,0], data[:,1], data[:,2],c=np.mean(signal,axis=1),s=10, vmax=vmax, vmin=vmin)

    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,100)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    

    plt.colorbar(sc)
    
    # plt.tight_layout()
    plt.savefig(fname)
    plt.close()