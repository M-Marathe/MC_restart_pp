import numpy as np
import matplotlib.pyplot as plt
import random


def gaussian_dis(x, mu, sigma):
    gaussian = np.exp(-((x - mu) / sigma) ** 2 / 2) / np.sqrt(2 * np.pi * sigma ** 2)
    return gaussian


random.seed(12345)
bin_size = 10
x_range = np.linspace(-180, 180, bin_size)
print(x_range, x_range.size)
sigma = bin_size*10
x_data = random.sample(range(-180, 180), 10)
print(x_data)
p_mu = np.zeros([len(x_data), bin_size])
ind1 = 0
for value in x_data:
    ind2 = 0
    total = 0
    for mu in x_range:
        p_mu[ind1, ind2] = gaussian_dis(value, mu, sigma)
        total = total + p_mu[ind1, ind2]
        ind2 += 1
    ind1 += 1
    print(total)
np.set_printoptions(precision=4, suppress=True)
print(p_mu)
'''
Now the next question is how to normalize to 1.0 
for each value in x_data? 
--
And also to incorporate it into my other script!! 
'''