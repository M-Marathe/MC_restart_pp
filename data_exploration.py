import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
#import random
from pathlib import Path
from scipy.stats import pmean #import moment #import norm


def chk_size(restart_file):
    '''
    Reading header of restart file to obtain the system size
    and number of ensembles used in MC simulations.
    Returns the number of sites, ensembles and atoms
    '''
    system_size, n_ens = 0, 0
    f = open(restart_file, 'r')
    for line in f:
        if 'atoms' in line:
            system_size = int(line.split()[-1])
        if 'ensemble' in line:
            n_ens = int(line.split()[-1])
            break
    f.close()
    n_atoms = np.cbrt(system_size)
    if n_atoms > int(n_atoms):
        sys.exit('**Error: Restart file might be corrupt**')
#    print("Number of atoms : ", n_atoms)
    return int(system_size), int(n_ens), int(n_atoms)


def m_coord(restart_file, size, ens):
    m_all = np.genfromtxt(restart_file, usecols=[4, 5, 6])#, max_rows=size)
    print(type(m_all), m_all.shape)
    m_carte = np.zeros((ens, size, 3))
    for i in range(ens):
        m_carte[i, :, :] = m_all[i*size:(i+1)*size, :]
        print(i, 'th ensemble   Max    Min')
        print('M_x  ', m_carte[i, :, 0].max(), m_carte[i, :, 0].min())
        print('M_y  ', m_carte[i, :, 1].max(), m_carte[i, :, 1].min())
        print('M_z  ', m_carte[i, :, 2].max(), m_carte[i, :, 2].min())
    return m_carte


def plot_momdist(restart_file, title='', figfile='', NDIV=36 * 2, cmap='Blues'):
    '''
    (( This function copied from Rafael's script ))
    Plots (and saves) radial histogram of spins distribution from a restart.*.out file

    Input:
    restart_file (str): path to UppASD output restart.*.out  file.
    title (str): Optional, adds a title to figure.
    figfile (str): Optional, path to save figure.
    NDIV (int): Optional, number of bins to divide the radial angle.
    cmap (str): Opional, matplolib colour map to be used.
    '''

    raw_data = np.genfromtxt(restart_file, usecols=[4, 5, 6])  # In cartesian coordinates

    # Convert to spherical coordinates accoding to the physical convention
    spherical = np.zeros(raw_data.shape)
    # Radius (not used, yet)
    spherical[:, 0] = np.sqrt(raw_data[:, 0] ** 2 + raw_data[:, 1] ** 2 + raw_data[:, 2] ** 2)
    # Theta
    spherical[:, 1] = np.arctan2(raw_data[:, 1], raw_data[:, 0])
    # Phi. for elevation angle defined from Z-axis down
    spherical[:, 2] = np.arctan2(np.sqrt(raw_data[:, 0] ** 2 + raw_data[:, 1] ** 2), raw_data[:, 2])

    # Split in intervals for plotting
    theta = np.histogram(spherical[:, 1], NDIV)
    phi = np.histogram(spherical[:, 2], int(NDIV / 2))

    width_theta = np.diff(theta[1])[0]
    width_phi = np.diff(phi[1])[0]

    cmap = plt.get_cmap(cmap)  # more in https://matplotlib.org/stable/tutorials/colors/colormaps.html

    # Centered normalization around the mean for the colours
    mean_theta = np.mean(theta[0])
    mean_phi = np.mean(phi[0])

    norm_theta = mcolors.CenteredNorm(vcenter=mean_theta)(theta[0])
    norm_phi = mcolors.CenteredNorm(vcenter=mean_phi)(phi[0])
    # is set halfrange to max(abs(A-vcenter))
    rgba_theta = cmap(norm_theta)
    rgba_phi = cmap(norm_phi)

    fig = plt.figure()

    # Polar plot
    ax1 = fig.add_subplot(121, polar=True)
    ax1.bar(theta[1][:-1], theta[0], width_theta, color=rgba_theta, align="edge", bottom=0.0, edgecolor="k")
    ax1.set_xlabel(r'$\theta$')

    # Remove  labels from radial ticks
    ax1.set_yticklabels([])
    # Sets the Zero on the top
    ax1.set_theta_direction(-1)
    # Changes to clockwise
    ax1.set_theta_offset(np.pi / 2.0)

    ax2 = fig.add_subplot(122, polar=True)

    ax2.bar(phi[1][:-1], phi[0], width_phi, color=rgba_phi, align="edge", bottom=0.0, edgecolor="k")
    ax2.set_xlabel(r'$\phi$')
    # Remove  labels from radial ticks
    ax2.set_yticklabels([])
    # Sets the Zero on the top
    ax2.set_theta_direction(-1)
    # Changes to clockwise
    ax2.set_theta_offset(np.pi / 2.0)
    # Limits angles
    ax2.set_thetalim(0, np.pi)

    if title != '':
        plt.figtext(0.5, 0.85, title, ha='center', va='center')

    if figfile != '':
        plt.savefig(figfile)

    plt.show()


def average_qtt(p):
    '''
    function to read cumulant file directly from the same directory as restart
    file and getting average magnetic moment. To use as a label for classification!
    Here I can modify to read instead Binder cumulant when AFM problem fixed
    to use as a label instead of M_avg.
    '''
    cumulant_files = list(p.glob('T10/cumu*out'))
    #if not cumulant_files:   #Equivalent to cumulant_files == [] as a condition
    #    sys.exit('No cumulant file present in the directory')
    m_avg = None
    for cum_file in cumulant_files:
        f = open(cum_file)
        contents = f.readlines()
        m_avg = contents[-1].split()[1]
    if not m_avg:
        sys.exit('**Error: Cumulant file absent or corrupt**')
    return m_avg


p = Path('.')
restart_files = list(p.glob('T10/restart*out'))
if_plot = False
for file in restart_files:
    system_size, n_ens, n_atoms = chk_size(file)
    print(system_size, n_ens, n_atoms)
    m_carte = m_coord(file, system_size, n_ens)
    print(m_carte.shape)
    if if_plot:
        plot_momdist(file, 'test', 'plot.ps')
m_avg = average_qtt(p)
print(m_avg)
#        print('#Component  Maximum   Minimum   Mean for abs values')
#        print('M_x  ', m_carte[:, 0].max(), m_carte[:, 0].min(), pmean(abs(m_carte[:, 0]), 1))
#        print('M_y  ', m_carte[:, 1].max(), m_carte[:, 1].min(), pmean(abs(m_carte[:, 1]), 1))
#        print('M_z  ', m_carte[:, 2].max(), m_carte[:, 2].min(), pmean(abs(m_carte[:, 2]), 1))
