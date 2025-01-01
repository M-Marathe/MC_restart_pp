import numpy as np
#
# 1. read restart file
# 2. Convert to only "numbers" and then only M-components
# 3. Read and store as a binary file
#
restart_path = './T400/'
file_name = restart_path+'restart.cubic_3d.out'
#data_file = open('M_compo.txt', 'w')
#print(file_name)
#restart_file = open(file_name, 'r')
#for line in restart_file:
#    print(line, end="")
#    if "#" not in line:
#        data_file.write(line)
#
#data_file.close()
M_compo = np.loadtxt(file_name, skiprows=7, usecols= (4,5,6))
print(type(M_compo), M_compo.dtype, M_compo.shape)

np.astype(M_compo, np.float64).tofile("M_compo.bin")
