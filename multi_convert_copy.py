import numpy as np
from pathlib import Path
#
# 1. Define an array of temperatures to read files from
# 2. Create a module (based on convert.py) to create a single binary
## file containing data for different T's and M_i's


def conversion(file_list):
#    restart_path = './T400/'
#    file_name = restart_path+'restart.cubic_3d.out'
 #   magn_compo = np.empty()
    for i in range(len(file_list)):
        file_name = file_list[i]
        magn_compo = np.loadtxt(file_name, skiprows=7, usecols= (4,5,6))
        print(type(magn_compo), magn_compo.dtype, magn_compo.shape)

 #       t_magn[i] = np.astype(magn_compo, np.float64).tolist()
        np.astype(magn_compo, np.float64).tofile("M_compo.bin")
 #   np.astype(magn_compo, np.float64).tofile("M_compo.bin")

    return


p = Path('.')
file_list = list(p.glob('T*/restart*'))
conversion(file_list)
