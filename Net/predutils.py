# full imports
import config

# aliases
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as pltc

# single imports
import numpy as np
from collections import Counter
from skimage.filters import threshold_otsu
from os import sep

# main imports
# full imports
import configparser
import pickle
import time

# aliases
import dataprocessing as dp
import siamese as s
import sklearn.metrics as skm


def spatial_correction(prediction, radius=3):
    """
    Function returning a copy of the prediction map with spatial correction. Each pixel is resampled
    with the most frequent class in a kernel with the given radius, surrounding the pixel. If there's a tie between
    the classes, the original label is kept.

    :param prediction: a 2-dim array containing the predicted classes
    :param radius: a positive integer indicating the radius of the "kernel", including the central pixel
                   (Default =3 => 5x5 kernel)

    :return: a copy of prediction with corrected labels
    """
    corrected = np.zeros(prediction.shape)
    max_r, max_c = prediction.shape
    for row in range(max_r):
        for col in range(max_c):
            upper_x = max(0, row - (radius - 1))
            upper_y = max(0, col - (radius - 1))
            # note: the lower bound for the moving "kernel" must be one unit greater for each coordinate than the
            # actual lower bound, since it will be discarded as the last index for the slices
            lower_x = min(max_r, row + radius)
            lower_y = min(max_c, col + radius)
            counter = Counter(prediction[upper_x:lower_x, upper_y:lower_y].ravel())
            counts = counter.most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                corrected[row, col] = prediction[row, col]
            else:
                corrected[row, col] = counts[0][0]
    return corrected


def plot_maps(prediction, label_map):
    """
    Function plotting the original label map put beside the predicted label map

    :param prediction: the 2-dim array of shape (height x width) of predicted classes
    :param label_map: the 2-dim array of shape (height x width) of labels loaded with the dataset

    :return: a matplotlib figure containing the map of the whole prediction, the same map with a mask covering unknown
             labels and the ground truth
    """

    new_map = np.copy(prediction)
    replace_indexes = np.where(label_map == config.UNKNOWN_LABEL)
    new_map[replace_indexes] = config.UNKNOWN_LABEL

    cmap = pltc.ListedColormap(config.COLOR_MAP)
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(prediction, cmap=cmap, vmin=0, vmax=2)
    ax1.title.set_text("Total prediction")

    ax2.imshow(new_map, cmap=cmap, vmin=0, vmax=2)
    ax2.title.set_text("Comparable Prediction")

    ax3.imshow(label_map, cmap=cmap, vmin=0, vmax=2)
    ax3.title.set_text("Ground truth")
    plt.show()
    return fig


def pseudo_labels(first_img, second_img, dist_function, return_distances=False):
    """
    Function generating the pseudo labels for a given image pair. The pseudo labels are generated by applying
    the distance function directly on the pair and then using Otsu thresholding.

    :param first_img: the first images of the pair. It is a 2-dim array of shape (height x width, values), generally
                      a slice of the output of dataprocessing.preprocessing with keep_unlabeled=True.
    :param second_img: the first images of the pair. It is a 2-dim array of shape (height x width, values), generally
                      a slice of the output of dataprocessing.preprocessing with keep_unlabeled=True
    :param dist_function: The function to be used for distance computation. SAM and euclidean_distance are the
                          ones implemented so far
    :param return_distances: A boolean flag indicating whether to return the map with distances (True) or labels (False)

    :return: a map of pseudo labels (or distances) as a 1-dim array with shape (height x width) and the threshold used
    """

    img_a = tf.constant(first_img)
    img_b = tf.constant(second_img)
    distances = dist_function((img_a, img_b)).numpy()
    threshold = threshold_otsu(distances)
    if return_distances is True:
        returned_map = distances
    else:
        returned_map = np.where(distances > threshold, config.CHANGED_LABEL, config.UNCHANGED_LABEL)
    return returned_map, threshold


def labels_by_percentage(pseudo_dict, percentage):
    """
    Function extracting the position of best pixel pairs according to their distance. The extraction is stratified with
    respect to the complete collection of distances, so that the resulting set would contain the percentage% of the
    closest pairs and the percentage% of the farthest pairs. In order to work correctly, the pairs must be in the
    same order of the given distances.

    :param pseudo_dict: a dictionary containing the pre-computed distances of the pairs, the threshold for
                        label-conversion and original shape of the images. Basically, it's the dumped result of
                        the "main" script
    :param percentage: float value in ]0,1] indicating the percentage of the best pairs to be extracted

    :return: a 1 dim array containing the position of extracted pixel pairs and a 1-dim containing the labels
            warning: the returned arrays are ordered so that the closest pairs are returned ordered before the farthest
            pairs (which are also ordered), so a shuffle before usage might be necessary
    """
    # check of percentage value
    if percentage <= 0 or percentage > 1:
        raise ValueError("ERROR: percentage must be a float in ]0,1]")

    pseudo_distances = pseudo_dict["distances"]
    threshold = pseudo_dict["threshold"]

    # selecting the indexes of the not-changed (N) pairs and of the changed ones (C)
    N = np.where(pseudo_distances <= threshold)
    C = np.where(pseudo_distances > threshold)

    # packing the distances of not-changed and changed pairs with the respective position in the array
    nmatrix = np.c_[pseudo_distances[N], N[0]]
    cmatrix = np.c_[pseudo_distances[C], C[0]]

    # ordering the values in ascending order for unchanged pairs and descending for changed ones
    # extreme values = more confidence in the respective labeling
    nmatrix = nmatrix[nmatrix[:, 0].argsort()]
    cmatrix = cmatrix[(-cmatrix)[:, 0].argsort()]

    # generating a new array of labels for the selected data.
    labels = np.concatenate((np.full(int(percentage*len(nmatrix)), config.UNCHANGED_LABEL),
                            np.full(int(percentage*len(cmatrix)), config.CHANGED_LABEL)))

    # concatenation of the selected pairs (first unchanged, then changed)
    # the selection is given by extracting the desired percentage of ordered indexes from each class
    selected_data = np.concatenate((nmatrix[:int(percentage*len(nmatrix)), 1].astype(int),
                                   cmatrix[:int(percentage*len(cmatrix)), 1].astype(int)))

    return selected_data, labels


def labels_by_neighborhood(pseudo_dict, radius=3):
    """
    Function extracting the position of the best pixel pairs according to their neighborhood.
    The extraction is performed by excluding the pairs surrounded with at least one label different from the one
    assigned to them. The distances are converted to labels, reshaped and spatial corrected before the extraction.
    In order to work correctly, the pairs must be in the same order of the labels.

    :param pseudo_dict: a dictionary containing the pre-computed distances of the pairs, the threshold for
                        label-conversion and original shape of the images. Basically, it's the dumped result of
                        the "main" script
    :param radius: integer value indicating the radius of the square patch of the neighbouring pixels

    :return: a 1 dim array containing the position of the extracted pixel pairs and a 1-dim containing the labels

    """

    if radius < 1:
        raise ValueError("ERROR: radius must be a int >=1")

    # converting distances in labels and performing correction
    pseudo_lab = np.where(np.reshape(pseudo_dict["distances"], pseudo_dict["shape"]) > pseudo_dict["threshold"],
                                     config.CHANGED_LABEL, config.UNCHANGED_LABEL)
    pseudo_lab = spatial_correction(np.reshape(pseudo_lab, pseudo_dict["shape"]))

    selected_data = []
    label_list = []
    max_r, max_c = pseudo_lab.shape
    for row in range(max_r):
        for col in range(max_c):
            upper_x = max(0, row - (radius - 1))
            upper_y = max(0, col - (radius - 1))
            # note: the lower bound for the moving "kernel" must be one unit greater for each coordinate than the
            # actual lower bound, since it will be discarded as the last index for the slices
            lower_x = min(max_r, row + radius)
            lower_y = min(max_c, col + radius)
            counter = Counter(pseudo_lab[upper_x:lower_x, upper_y:lower_y].ravel())
            counts = counter.most_common()
            if len(counts) == 1:
                selected_data.append(row*max_c + col)
                label_list.append(pseudo_lab[row, col])
    return np.asarray(selected_data), np.asarray(label_list)


"""
    Script for the generation of pseudo labels as a dictionary with 3 entries:
        - 'distances': a 1-dim array of length (heightXwidth) containing the distances between each pair
        - 'threshold': a float value to be used as threshold when getting the labels from "distances". It is obtained
                      with the otsu method
        - 'shape': a tuple containing the original shape of the image as (height, width)
    The pseudo labels are saved in the path indicated in the dedicated section of the selected dataset in net.conf 
    (pseudoPath).
    The script also generates a .csv file containing the confusion matrices and the accuracy computed before and after
    the spatial correction and the time elapsed during computation and correction, and a .png file containing the plot
    of the pseudo labels (see "plot_maps" function). These files are saved in the path indicated in config.py.
"""
if __name__ == '__main__':
    # opening the settings file
    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    # getting dataset name, rescaling option and distance function
    dataset = parser["settings"].get("train_set")

    rescaling = parser["settings"].getboolean("apply_rescaling")

    if parser["settings"].get("distance") == "ED":

        distance_func = s.euclidean_dist
    elif parser["settings"].get("distance") == "SAM":

        distance_func = s.SAM
    else:

        raise NotImplementedError("Error: DISTANCE FUNCTION NOT IMPLEMENTED")

    print("Info: STARTING PSEUDO LABEL GENERATION FOR " + dataset + " WITH " + parser["settings"].get("distance")
          + " AND RESCALING=" + str(rescaling))

    # loading and processing the dataset
    img_a, img_b, labels, names = dp.load_dataset(dataset, parser)
    processed_ab, processed_lab = dp.preprocessing(img_a, img_b, labels, parser[dataset],
                                                   keep_unlabeled=True,
                                                   apply_rescaling=rescaling)
    i = 0
    for lab in labels:
        # selecting a image from the list of pairs
        pro_a = processed_ab[i:i+lab.size, 0]
        pro_b = processed_ab[i:i+lab.size, 1]
        pro_lab = processed_lab[i:i+lab.size]

        # generating distances
        print("Info: GENERATING DISTANCES OF " + names[i] + " " + str(i+1) + "/" + str(len(labels)))
        tic = time.time()
        dist, thresh = pseudo_labels(pro_a, pro_b, distance_func, return_distances=True)
        toc = time.time()

        # dumping values in a file
        print("Info: SAVING DISTANCES OF " + names[i] + " " + str(i+1) + "/" + str(len(labels)))
        dist_file = open(parser[dataset].get("pseudoPath") + sep + names[i] + ".pickle", "wb")
        pickle.dump({'threshold': thresh, 'distances': dist, 'shape': lab.shape}, dist_file, pickle.HIGHEST_PROTOCOL)
        dist_file.close()

        print("Info: COMPUTING METRICS AND MAP PLOT...")
        pseudo = np.where(dist > thresh, config.CHANGED_LABEL, config.UNCHANGED_LABEL)

        cm = skm.confusion_matrix(pro_lab, pseudo, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

        metrics = s.get_metrics(cm)

        file = open(config.STAT_PATH + dataset + "_" + names[i] + "_" + parser["settings"].get("distance")
                    + "_pseudo_rescaling_" + str(rescaling) + ".csv", "w")

        # printing columns names, number of examples and threshold used
        file.write("total_examples, threshold")

        for k in metrics.keys():
            file.write(", " + k)
        file.write(", time")

        for k in metrics.keys():
            file.write(", " + k + "_correction")
        file.write(", time_correction")
        file.write("\n" + str(len(pro_lab)) + ", " + str(thresh))

        # printing metrics
        for k in metrics.keys():
            file.write(", " + str(metrics[k]))
        file.write(", " + str(toc-tic))

        # saving the map plot
        pseudo_map = np.reshape(pseudo, lab.shape)
        ground_t = dp.refactor_labels(lab, parser[dataset])
        fig = plot_maps(pseudo_map, ground_t)
        fig.savefig(config.STAT_PATH + dataset + "_" + names[i] + "_" + parser["settings"].get("distance")
                    + "_pseudo_rescaling_" + str(rescaling) + ".png",
                    dpi=300, bbox_inches='tight')

        print("Info: EXECUTING SPATIAL CORRECTION AND COMPUTING METRICS...")
        # spatial correction
        tic = time.time()
        corrected_map = spatial_correction(pseudo_map)
        toc = time.time()

        # metrics
        sccm = skm.confusion_matrix(pro_lab, corrected_map.ravel(),
                                    labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
        scmetrics = s.get_metrics(sccm)

        # saving the metrics
        for k in scmetrics.keys():
            file.write(", " + str(scmetrics[k]))
        file.write(", " + str(toc - tic))
        file.write("\n")
        file.close()

        # map plotting
        scfig = plot_maps(corrected_map, ground_t)
        scfig.savefig(config.STAT_PATH + dataset + "_" + names[i] + "_" + parser["settings"].get("distance")
                      + "_pseudo_rescaling_" + str(rescaling) + "_corrected.png", dpi=300, bbox_inches='tight')

        i = i+lab.size
