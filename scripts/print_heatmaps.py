import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import sys
import matplotlib
matplotlib.use('Agg')
from utils.helper import heatmap
import matplotlib.pyplot as plt


if __name__ == "__main__":
        
    start_class = int(sys.argv[1])
    end_class = int(sys.argv[2])
    network = sys.argv[3]
    method = sys.argv[4]

    classes = np.arange(start_class, end_class)
    num_per_class = 50
    
    source_dir = os.path.join("maps", "imagenet")
    store_dir = os.path.join("maps", "imagenet_png")
        
    for c in tqdm(classes):
        start = c * num_per_class
        end = start + num_per_class

        for i in range(num_per_class):
            if os.path.isfile(os.path.join(source_dir, "%d_%d_%s_%s.npz"%(c, i, network, method))):
                data = np.load(os.path.join(source_dir, "%d_%d_%s_%s.npz"%(c, i, network, method)))
                heatmap(data['x'])
                data.close()
                plt.savefig(os.path.join(store_dir, "%d_%d_%s_%s.png"%(c, i, network, method)))
                plt.close()