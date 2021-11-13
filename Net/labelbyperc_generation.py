# full imports
import config
import configparser
import pickle
import os

# aliases
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc

# single import
from predutils import labels_by_percentage

"""
    Script for generating the plot of the best pseudo-label extracted by percentage. 
    Plot are generated from 10% to 90%, with a 10% difference between to subsequent plots.
    The script uses the function "labels_by_percentage" from predutils.py in order to get the changed pixels
        ordered by descending distance and the unchanged pixel by ascending distance.
    Each plot contains a map of the extracted "Changed labels" (C label), a map of the extracted "Unchanged labels"
        (N label) and a map containing both (NC label) and is saved in the path specified in config.py
"""

if __name__ == '__main__':
    # opening the settings file
    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    # getting the name of the set of pseudo-labels to be used
    dataset = parser["settings"].get("train_set")

    for image_name in os.listdir(parser[dataset].get("pseudoPath")):
        # loading the pseudo-labels dictionary for each image
        print("Info: GENERATING " + os.path.splitext(image_name)[0] + " MAP...")
        pseudo = open(parser[dataset].get("pseudoPath") + os.sep + image_name, "rb")
        dict = pickle.load(pseudo)
        pseudo.close()

        # getting pixels from each class ordered by distance
        best_data, labels = labels_by_percentage(dict, percentage=1)

        # separating the position of changed pairs from the unchanged ones
        cex = best_data[np.where(labels == config.CHANGED_LABEL)]
        nex = best_data[np.where(labels == config.UNCHANGED_LABEL)]
        for i in range(1, 10):
            print("Info: percentage " + str(i) + "0%")

            # generating the empty maps
            nmap = np.full(dict["shape"][0] * dict["shape"][1], fill_value=2)
            cmap = np.full(dict["shape"][0] * dict["shape"][1], fill_value=2)
            ncmap = np.full(dict["shape"][0] * dict["shape"][1], fill_value=2)

            # populating the maps
            nmap[nex[:int(len(nex)*(i/10))]] = config.UNCHANGED_LABEL
            ncmap[nex[:int(len(nex)*(i/10))]] = config.UNCHANGED_LABEL
            cmap[cex[:int(len(cex)*(i/10))]] = config.CHANGED_LABEL
            ncmap[cex[:int(len(cex)*(i/10))]] = config.CHANGED_LABEL

            nmap = np.reshape(nmap, dict["shape"])
            cmap = np.reshape(cmap, dict["shape"])
            ncmap = np.reshape(ncmap, dict["shape"])

            # plotting the maps
            color_map = pltc.ListedColormap(config.COLOR_MAP)
            fig = plt.figure(figsize=(16, 9))
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)
            ax1.imshow(cmap, cmap=color_map, vmin=0, vmax=2)
            ax1.title.set_text("C label")

            ax2.imshow(nmap, cmap=color_map, vmin=0, vmax=2)
            ax2.title.set_text("N label")

            ax3.imshow(ncmap, cmap=color_map, vmin=0, vmax=2)
            ax3.title.set_text("NC label")

            fig.savefig(config.STAT_PATH + dataset + "_" + os.path.splitext(image_name)[0] + "_" + str(i) + "0%.png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
