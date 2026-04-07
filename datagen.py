import random
import numpy as np
N = 10 
L = 100
M = 5

# numpy
points = np.random.uniform(-L,L,size=(N,3))
mass = np.random.uniform(0,M,size=(N,1))
bodys = np.hstack([points, mass])
np.savetxt("tests/test0.txt", bodys, fmt="%.6f", delimiter=" ")