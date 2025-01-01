import numpy as np


def binning():
    arr_1 = [0, 180]
    arr_2 = [-180, 0, 180]
#    theta_r = [0]
#    phi_r = [0]
#    for i in range(1, 181):
#        theta_r.append(i)
#        phi_r.append(-i)
#        phi_r.append(i)
#    phi_r.sort()
    x = np.histogram_bin_edges(arr_1, bins=10, range=(0, 180))
    y = np.histogram_bin_edges(arr_2, bins=20, range=(-180, 180))
    print('I am here')
    print(np.digitize(arr_1, x, right=True))
    print(np.digitize(arr_2, y, right=True))
    #print(theta_bin)
    #print(phi_bin)
    print(x, y)

#    print(len(theta_r), len(phi_r))
#    print(theta_r[0], theta_r[180])
#    print(phi_r[0], phi_r[180], phi_r[360])

binning()
m_carte = np.genfromtxt('trial.dat', usecols=[4, 5, 6])
print(m_carte.shape)
#print(m_carte)
m_spher = np.zeros(m_carte.shape)
# Coordinate conversion
# Radius
m_spher[:, 0] = np.sqrt(m_carte[:, 0]**2 + m_carte[:, 1]**2 + m_carte[:, 2]**2)
# Polar angle Theta
m_spher[:, 1] = np.arctan2(np.sqrt(m_carte[:, 0]**2 + m_carte[:, 1]**2), m_carte[:, 2]) * 180 / np.pi
 # Azimuthal angle Phi
m_spher[:, 2] = np.arctan2(m_carte[:, 1], m_carte[:, 0]) * 180 / np.pi
print(m_spher.shape)
#print(m_spher)
c_array = np.zeros([13, 3, 1])
#print(c_array.shape)
c_array = np.concatenate((m_carte, m_spher), axis=0)
v_array = np.vstack((m_carte, m_spher))
h_array = np.hstack((m_carte, m_spher))
s_array = np.stack((m_carte, m_spher))
print(c_array.shape, v_array.shape, h_array.shape)
print(s_array.shape)

