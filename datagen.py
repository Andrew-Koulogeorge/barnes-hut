import random
import numpy as np

L = 1000
M = 100

# numpy
for N in [10,100,500,1_000, 5_000, 10_000, 25_000, 50_000, 500_000]:
    points = np.random.uniform(-L,L,size=(N,3))
    mass = np.random.uniform(M/2,M,size=(N,1))
    bodys = np.hstack([points, mass])
    with open(f"test/test_traces/test_{N}.txt", "w") as f:
        np.savetxt(f, bodys, fmt="%.6f", delimiter=" ")