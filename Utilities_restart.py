import numpy as np
import argparse
import sys
import random
from pathlib import Path
#import os


def chk_size(restart_file):
    '''
    Reading header of restart file to obtain the system size.
    This is where one can add lines if ensemble size also
    is needed and to be used.
    '''
    system_size = 0
    f = open(restart_file, 'r')
    for line in f:
        if 'atoms' in line:
            system_size = int(line.split()[-1])
            break
    f.close()
#    n_atoms = np.cbrt(system_size)
    return int(system_size)


def coord_conv(restart_file, size, n_bins):
    '''
    Function to convert coordinate system of magnetic components.
    Reads each file separately and returns 2 ndarrays containing
    theta and phi. It reads only first ensemble data determined
    by system size. May need to adjust if we want to consider
    all ensembles from Monte Carlo simulations.
    '''
    # Read magnetic moment components in Cartesian coords
    m_carte = np.genfromtxt(restart_file, usecols=[4, 5, 6], max_rows=size)
    # Storage of spherical coordinates
    m_spher = np.zeros(m_carte.shape)
    # Coordinate conversion
    # Radius
    m_spher[:, 0] = np.sqrt(m_carte[:, 0]**2 + m_carte[:, 1]**2 + m_carte[:, 2]**2)
    # Polar angle Theta
    m_spher[:, 1] = np.arctan2(np.sqrt(m_carte[:, 0]**2 + m_carte[:, 1]**2), m_carte[:, 2]) * 180 / np.pi
    # Azimuthal angle Phi
    m_spher[:, 2] = np.arctan2(m_carte[:, 1], m_carte[:, 0]) * 180 / np.pi
    # Create equal spaced arrays for digitize function to use
    theta = np.linspace(0,180, n_bins)
    phi = np.linspace(-180, 180, n_bins*2)
#    print(type(theta), type(phi))
#    print(theta.shape, phi.shape)
#   print(theta, phi)
    digit_theta = np.digitize(m_spher[:, 1], theta, right=True)
    digit_phi = np.digitize(m_spher[:, 2], phi, right=True)
#    print(len(theta), len(phi))
#    print(digit_theta.shape)
#    print(digit_phi.shape)
#    print(m_spher[:, 1])
#    print(digit_theta)
#    print(m_spher[:, 2])
#    print(digit_phi)
    return digit_theta, digit_phi


def label_cond(n_images, avg_file):
    lbl_m = np.zeros(n_images)   # in the main file use another size variable here
    temperature = np.zeros(n_images)   # and here + condition at the end to check mismatch
    t_index = 0
    f = open(avg_file, 'r')
    for line in f:
        if t_index >= n_images:
            sys.exit('**ERROR: More T-points in averaged data**')
        if line[0] != '#':
            total_m = float(line.split()[1])
            temperature[t_index] = float(line.split()[0])
            if total_m < 0.4:       # paramagnetic (HT)
                lbl_m[t_index] = 0
            elif total_m >= 0.4 and total_m <= 0.5:    # around T_c
                lbl_m[t_index] = 1
            else:                   # ferromagnetic (LT)
                lbl_m[t_index] = 2
            t_index += 1
    if t_index < n_images:
        sys.exit('**ERROR: Fewer T-points in averaged data**')
    if temperature[-1] < temperature[0]:
        lbl_m = np.flip(lbl_m)
    return lbl_m


def rand_array(n_images, train_data_per):
    random.seed(12345)
    train_size = round(n_images/100*train_data_per)
#    print(train_size)
    train_index = random.sample(range(n_images), k=train_size)
#    print(train_data_per, train_index)
#    print(type(train_index), len(train_index))
    return train_size, train_index


def binary_conv(file_list, avg_file, n_bins, train_data_per):
    tot_size = 0
    size = 0      # Number of sites in a given file
    for i in range(len(file_list)):
        file_name = file_list[i]
        size = chk_size(file_name)
        tot_size = tot_size + size
    if tot_size != size * len(file_list):
        sys.exit('Err: System size differs in different files?')
    n_images = len(file_list)   # No. of images (Temperature data points)
    lbl_m = label_cond(n_images, avg_file)
#    print(lbl_m.size, lbl_m)
    print('No. of Temperatures in input: ', n_images)
    print('System size for each T: ', size)
    train_size, train_index = rand_array(n_images, train_data_per)
    print('Size of test dataset: ', train_size, '& training dataset: ', n_images - train_size)
    magn_compo_train = np.zeros([train_size, size, 2])             # Training dataset
    magn_compo_test = np.zeros([n_images - train_size, size, 2])   # Test dataset
    lbl_train = np.zeros([train_size, 1])            # Training labels
    lbl_test = np.zeros([n_images - train_size, 1])  # Test labels
    j1, j2 = int(0), int(0)
    for i in range(n_images):
        file_name = file_list[i]
        #print(file_name)
        if i in train_index:
            magn_compo_train[j1, :, 0], magn_compo_train[j1, :, 1] = coord_conv(file_name, size, n_bins)
            lbl_train[j1, :] = lbl_m[i]
            j1 = j1 + 1
        else:
            magn_compo_test[j2, :, 0], magn_compo_test[j2, :, 1] = coord_conv(file_name, size, n_bins)
            lbl_test[j2, :] = lbl_m[i]
            j2 = j2 + 1
#    print(j1, j2)
#    print(magn_compo_train.shape, magn_compo_test.shape)
#    print(magn_compo)
#    print(lbl_train)
#    print(lbl_test)
    train_restart = np.zeros([train_size, size, n_bins*n_bins*2])
    test_restart = np.zeros([n_images - train_size, size, n_bins*n_bins*2])
    for i in range(train_size):
        for j in range(size):
                m, n = int(magn_compo_train[i, j, 0]), int(magn_compo_train[i, j, 1])
                train_restart[i, j, m + n_bins*n] = 1
    for i in range(n_images - train_size):
        for j in range(size):
            m, n = int(magn_compo_test[i, j, 0]), int(magn_compo_test[i, j, 1])
            test_restart[i, j, m + n_bins*n] = 1
#    print(np.sum(train_restart))
#    print(np.sum(test_restart))
#    print(train_restart.shape, test_restart.shape)
    train_restart.astype(np.float32).tofile("train_restart.bin")
    test_restart.astype(np.float32).tofile("test_restart.bin")
    lbl_train.astype(np.float32).tofile("train_label.bin")
    lbl_test.astype(np.float32).tofile("test_label.bin")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To convert coordinates from restart file')
    parser.add_argument('--dir_path', type=str, default='.', help='Path to restart files')
    #parser.add_argument('restart_file', type=str, help='UppASD output restart.*.out  file')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins (for 180 degrees)')
    parser.add_argument('--train_data_per', type=float, default=60.0, help='Percentage data used for training ')
    args = parser.parse_args()
    p = Path(args.dir_path)
    file_list = sorted(list(p.glob('T1*/restart*.out')))
    avg_file = list(p.glob('thermal.dat'))
    # Format for thermal.dat:
    # Temp.   Mavg     UBinder    Susc.      Cv
    '''
    Here I am going to use already created file to read total average magnetic 
    moments at each T, but this requires many additional checks for whether T 
    is increasing/decreasing in this file, if it contains data for the same 
    run as was used for restart files, etc. More efficient might be to use 
    directly cumulant files within the same directory as restart files and 
    follow similar approach used in shell script for getting thermal.dat.
    TO BE DONE AT SOME LATER POINT. 
    '''
    print("Directory path: ", args.dir_path)
    print("No. of bins used: ", args.n_bins)
    print("Data used for training: ", args.train_data_per, "%")
    binary_conv(file_list, avg_file[0], args.n_bins, args.train_data_per)
