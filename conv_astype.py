import numpy as np
import matplotlib.pyplot as plt
#
arr_i = np.array([1.5e-2, 2, 3, 5, 10, 1/2])
print("i/p: ", arr_i)
#
np.astype(arr_i, np.float64).tofile("flt_arr.bin")

#file = open("flt_arr.bin", 'r', encoding='utf-16-le')
#print(file)
#
#with open('flt_arr.bin', 'r', encoding='utf-16-le') as file:
#    print(file)

with open('flt_arr.bin', 'rb') as f:
    # Read the data into a NumPy array
    array = np.fromfile(f, dtype=float)
print("o/p: ", array)