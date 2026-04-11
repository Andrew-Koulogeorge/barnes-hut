import random
import numpy as np
import pandas as pd

L = 10_000
M = 100

# numpy
for N in [10,100,500,1_000, 5_000, 10_000, 25_000, 50_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]:
    print(N)
    points = np.random.uniform(-L,L,size=(N,3))
    mass = np.random.uniform(M/2,M,size=(N,1))
    bodys = np.hstack([points, mass])
    with open(f"test/test_traces/test_{N}.txt", "w") as f:
        pd.DataFrame(bodys).to_csv(f"test/test_traces/test_{N}.txt", sep=" ", header=False, index=False, float_format="%.6f")
