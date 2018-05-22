"""
Histogram of turns (for future balancing of data)
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

from data_collection.data_collect import path

n_bins = [x - 0.5 for x in range(-10, 12)]

data = h5py.File(path, 'r')

fig, axs = plt.subplots()
axs.hist([d[1] for d in data['controls'][:]], bins=n_bins)

data.close()
plt.xticks(np.arange(-10, 11, step=1))
plt.show()
