import random
import numpy as np

random.seed(1)
size = 10
# 60% of population for training
#print(size/100*60)
train_size = round(size/100*60)
#print(train_size)
p = random.choices(range(size), k=train_size)
#print(p)
#print(type(p), len(p))
q = random.sample(range(size), k=train_size)
print(q)
#print(type(q), len(q))


