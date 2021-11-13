# full imports
import configparser
import config
import os
import pickle

# aliases
import numpy as np
import matplotlib.colors as pltc
import matplotlib.pyplot as plt

# single import
from predutils import labels_by_neighborhood

"""
    Script for generating the plot of the best pseudo-label extracted by neighbourhood. 
    Plot are generated from radius 2 to a maximum radius specified in net.conf. 
    The script uses the function "labels_by_neighbourhood" from predutils.py for the extraction.
    Each plot contains a map of the extracted "Changed labels" (C label), a map of the extracted "Unchanged labels"
        (N label) and a map containing both (NC label) and is saved in the path specified in config.py
"""
if __name__ == '__main__':

    # opening the settings file
    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    # getting the name of the set of pseudo-labels to be used and the maximum radius for the extraction
    dataset = parser["settings"].get("train_set")
    radius_max = int(parser["settings"].get("pseudo_radius"))

    for image_name in os.listdir(parser[dataset].get("pseudoPath")):
        # loading the pseudo-labels dictionary for each image
        print("Info: GENERATING " + os.path.splitext(image_name)[0] + " MAP...")
        pseudo = open(parser[dataset].get("pseudoPath") + os.sep + image_name, "rb")
        dict = pickle.load(pseudo)
        pseudo.close()

        for i in range(2, radius_max + 1):
            # extracting the labels
            print("Info: RADIUS " + str(i) + "/" + str(radius_max))
            best_data, labels = labels_by_neighborhood(dict, radius=i)
            print(len(best_data))
            # generating the three maps to be plotted
            nmap = np.full(dict["shape"][0]*dict["shape"][1], fill_value=2)
            cmap = np.full(dict["shape"][0]*dict["shape"][1], fill_value=2)
            ncmap = np.full(dict["shape"][0]*dict["shape"][1], fill_value=2)

            nmap[best_data[np.where(labels == config.UNCHANGED_LABEL)]] = config.UNCHANGED_LABEL
            cmap[best_data[np.where(labels == config.CHANGED_LABEL)]] = config.CHANGED_LABEL
            ncmap[best_data] = labels

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

            fig.savefig(config.STAT_PATH + dataset + "_" + os.path.splitext(image_name)[0] + "_radius" + str(i) + ".png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
