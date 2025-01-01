import numpy as np
import pandas as pd
from pathlib import Path


def create_bins(low_bound, upp_bound, bin_n):
    bins =[]
    width = (upp_bound - low_bound)/bin_n
    #for low in range(low_bound, low_bound + bin_n*width + 1, bin_n):
    #    bins.append((low, low+bin_n))
    for i in range(bin_n):
        bins.append((low_bound + i*width, low_bound + (i+1)*width))
    return bins

bins = create_bins(-1, 1, 20)
#print(bins)
bins2 = pd.IntervalIndex.from_tuples(bins)
#print(bins2)

p = Path('.')
file_list = list(p.glob('T400/restart*'))

with open(file_list[0], "rb") as f:
    num_lines = sum(1 for _ in f)
magn_compo = np.empty([len(file_list), 12288, 3])
for i in range(len(file_list)):
    file_name = file_list[i]
    magn_compo[i] = np.loadtxt(file_name, skiprows=7, usecols=(4, 5, 6))
    print(type(magn_compo), magn_compo.dtype, magn_compo.shape)
#    print(magn_compo.max(), magn_compo.min())

#magn_int = pd.cut(magn_compo, bins2)
#print(magn_int)