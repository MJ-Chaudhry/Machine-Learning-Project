import numpy as np
from scipy.stats import loguniform, uniform, randint

RI = randint(1, 10)
ri = RI.rvs()
ri_norm = RI.cdf(ri)

print(RI.dist.name)
print(ri, ri_norm, int(RI.ppf(0.567)))

kernels = ["linear", "rbf", "poly"]

U = uniform(0, 1)

# (i + 0.5)/len(kernels) = value

for i in range(len(kernels)):
    print(i, kernels[i], (i + 0.5)/len(kernels))

counts = {}
for i in range(10_000):
    value = U.rvs()
    idx = int(np.clip(np.floor(value * len(kernels)), 0, len(kernels) - 1))
    # print(value, idx, kernels[idx])
    if kernels[idx] not in counts:
        counts[kernels[idx]] = 0
    counts[kernels[idx]] += 1

print(counts)