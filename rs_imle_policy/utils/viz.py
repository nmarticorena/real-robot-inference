import numpy as np
import matplotlib.pyplot as plt


def colormap(distance, min_distance, max_distance, cmap="jet"):
    cmap = plt.get_cmap(cmap)
    norm_distance = (distance - min_distance) / (max_distance - min_distance)
    norm_distance = np.clip(norm_distance, 0, 1)
    colors = cmap(norm_distance)
    return colors
