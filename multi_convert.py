import numpy as np
import pandas as pd
from pathlib import Path
#
# 1. Define an array of temperatures to read files from
# 2. Create a module (based on convert.py) to create a single binary
## file containing data for different T's and M_i's


def conversion(file_list):
    magn_compo = np.empty([len(file_list), 12288, 3])  # Have set it by hand how to adjust it?
    for i in range(len(file_list)):
        file_name = file_list[i]
        magn_compo[i] = np.loadtxt(file_name, skiprows=7, usecols= (4,5,6))
#        print(magn_compo.max(),magn_compo.min())
        #print(type(magn_compo), magn_compo.dtype, magn_compo.shape)

#       t_magn[i] = np.astype(magn_compo, np.float64).tolist()
#       np.astype(magn_compo, np.float64).tofile("M_compo.bin")
#    print(type(magn_compo), magn_compo.dtype, magn_compo.shape)
    np.astype(magn_compo, np.float64).tofile("M_compo.bin")
    return


p = Path('.')
file_list = list(p.glob('T*/restart*'))
#with open(file_list[1], "rb") as f:
#    num_lines = sum(1 for _ in f)
#print(num_lines)

conversion(file_list)
