import numpy as np
import argparse
import sys
import random
from pathlib import Path
'''
This is to be included in the main Utilities!! 
It requires thermal.dat file which stores average moments and will need to 
contain 2 conditions:
1. a check for number of temperature folders with restart files matches with those 
in thermal.dat file.
2. 
'''
####
# Let's not worry about AFM... for now just using M_avg for labels!!
####


def label_cond(avg_file):
    lbl_m = np.zeros(36)   # in the main file use another size variable here
    temperature = np.zeros(36)   # and here + condition at the end to check mismatch
    t_index = 0
    f = open(avg_file, 'r')
    for line in f:
        if line[0] != '#':
            total_m = float(line.split()[1])
            temperature[t_index] = float(line.split()[0])
            if total_m < 0.4:
                lbl_m[t_index] = 0
            elif total_m >= 0.4 and total_m <= 0.5:
                lbl_m[t_index] = 1
            else:
                lbl_m[t_index] = 2
            t_index += 1
    print(lbl_m)
    print(temperature)
    if temperature[-1] < temperature[0]:
        lbl_m = np.flip(lbl_m)
    print(lbl_m)
    print(t_index)   # this number should match with other size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To convert coordinates from restart file')
    parser.add_argument('--dir_path', type=str, default='.', help='Path to restart files')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins (for 180 degrees)')
    parser.add_argument('--train_data_per', type=float, default=60.0, help='Percentage data used for training ')
    args = parser.parse_args()
    p = Path(args.dir_path)
    file_list = list(p.glob('T1*/restart*.out'))
    avg_file = 'thermal_full.dat'
# Format for thermal.dat:
# Temp.   Mavg     UBinder    Susc.      Cv
    print("Directory path: ", args.dir_path)
    label_cond(avg_file)
