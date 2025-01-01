import numpy as np
import argparse


def momconv(restart_file, n_bins):
    # Read magnetic moment components in Cartesian coords
    m_carte = np.genfromtxt(restart_file, usecols=[4, 5, 6])
    # Storage of spherical coordinates
    m_spher = np.zeros(m_carte.shape)
    # Coordinate conversion
    # Radius (should be one)
    m_spher[:, 0] = np.sqrt(m_carte[:, 0]**2 + m_carte[:, 1]**2 + m_carte[:, 2]**2)
    # Polar angle Theta
    m_spher[:, 1] = np.arctan2(np.sqrt(m_carte[:, 0]**2 + m_carte[:, 1]**2), m_carte[:, 2]) * 180 / np.pi
    # Azimuthal angle Phi
    m_spher[:, 2] = np.arctan2(m_carte[:, 1], m_carte[:, 0]) * 180 / np.pi
#    print('theta = ', m_spher[:, 1])
#    print('phi = ', m_spher[:, 2])
    # make histogram of equal width (will define my integer value)
    hist_t, theta = np.histogram(m_spher[:, 1], n_bins)
    hist_p, phi = np.histogram(m_spher[:, 2], n_bins*2)
    digit_theta = np.digitize(m_spher[:, 1], theta)
    digit_phi = np.digitize(m_spher[:, 2], phi)
    #print(m_spher[:, 1].shape, m_spher[:, 2].shape)
    #print(digit_theta.shape, digit_phi.shape)
    #print('in convergence function ... ')
    print(digit_theta.shape, digit_phi.shape)
    print(digit_theta.min(), digit_theta.max())
    print(digit_phi.min(), digit_phi.max())
    return m_spher


def binning(m_spher, n_bins):
    # make histogram of equal width (will define my integer value)
    theta = np.histogram_bin_edges(m_spher[:, 1], n_bins - 1)
    phi = np.histogram_bin_edges(m_spher[:, 2], n_bins * 2 - 1)
    #print(theta.shape, type(theta), theta)
    #print(phi.shape, type(phi), phi)
    digit_theta = np.digitize(m_spher[:, 1], theta)
    digit_phi = np.digitize(m_spher[:, 2], phi)
    print('In binning')
    print(digit_theta.shape, digit_phi.shape)
    #print('theta  = ', digit_theta)
    #print('phi  = ', digit_phi)
    print(digit_theta.min(), digit_theta.max())
    print(digit_phi.min(), digit_phi.max())
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To convert coordinates from restart file')
    parser.add_argument('restart_file', type=str, help='UppASD output restart.*.out  file')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins (for 180 degrees)')
    args = parser.parse_args()
    m_spher = momconv(args.restart_file, args.n_bins)
    binning(m_spher, args.n_bins)
