import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


def chk_size(restart_file):
    system_size = 0
    f = open(restart_file, 'r')
    for line in f:
        if 'atoms' in line:
            system_size = int(line.split()[-1])
            break
    f.close()
    n_atoms = np.cbrt(system_size)
    return int(n_atoms)


def moments_plot(restart_file, layer_no, plot_plane):
    if plot_plane not in [0, 1, 2]:
        sys.exit('Wrong plane for plotting; for more info type --help')

    m_carte = np.genfromtxt(restart_file, usecols=[4, 5, 6])
    ## Only required for naming convention I use T*/restart.*out
    temperature = restart_file.split('/')[0]
    temperature = temperature.replace("T", "0")
    ##
    n_atoms = chk_size(restart_file)
    if layer_no >= n_atoms:
        sys.exit('Layer number exceeds system size: check i/p')
#    print(n_atoms, layer_no, plot_plane)
    X, Y = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
#    print(X.shape)
    fig, ax = plt.subplots()
    title = 'Local moments at ' + temperature + ' K for ' + str(layer_no) + '$^{th}$ layer'
    ax.set_title(title)
    index_s = layer_no * n_atoms * n_atoms
    index_f = (layer_no + 1) * n_atoms * n_atoms
    if plot_plane == 0:
        m, n = 0, 1
        ax.set_xlabel('M$_x$')
        ax.set_ylabel('M$_y$')
        pl_name = '2d_moments_' + temperature + 'K_xy'
    elif plot_plane == 1:
        m, n = 1, 2
        ax.set_xlabel('M$_y$')
        ax.set_ylabel('M$_z$')
        pl_name = '2d_moments_' + temperature + 'K_yz'
    else:
        m, n = 2, 1
        ax.set_xlabel('M$_z$')
        ax.set_ylabel('M$_x$')
        pl_name = '2d_moments_' + temperature + 'K_zx'
    # Plotting
#    fig, ax = plt.subplots()
    ax.quiver(X, Y, m_carte[index_s:index_f, m], m_carte[index_s:index_f, n])
    plt.savefig(pl_name)
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Layer-wise plot of local moments')
    parser.add_argument('restart_file', type=str, help='UppASD restart file run')
    parser.add_argument('--layer_no', type=int, default='4', help='Layer number for plotting')
    parser.add_argument('--plot_plane', type=int, default='0', help='0: xy-plane, 1: yz-plane, 2: zx-plane')
    args = parser.parse_args()
    moments_plot(args.restart_file, args.layer_no, args.plot_plane)
