# full imports
import PIL
import os
import config

# aliases
import numpy as np
import sklearn.preprocessing as sp
import scipy.io as sio
import pandas as pd

# single imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_mat(path, label):
    """
    Function returning an image read from a .mat file in "path".

    :param path: string containing the path of the image
    :param label: label corresponding to the image in che .mat dictionary

    :return: a 3-dim numpy array containing the image
    """
    mat = sio.loadmat(path)
    return mat[label]


def minmax_pair(img1, img2):
    """
    Function rescaling the pair of images with min-max scaling.
    It assumes img1 and img2 are same-sized 2-dim arrays with shape (height x width,spectral bands).

    :param img1: 2-dim numpy array containing pixels of the first images
    :param img2: 2-dim numpy array containing pixels of the second images

    :return: two 2-dim numpy array containing the rescaled pixel values
    """
    # concatenation
    toscale = np.concatenate((img1, img2), axis=0)
    scaler = sp.MinMaxScaler()

    # fitting and transforming
    scaler.fit(toscale)
    scaled = scaler.transform(toscale)

    # re-splitting images
    return np.split(scaled, 2)


def refactor_labels(labels, conf_section):
    """
    Function refactoring the label of a dataset setting "0" as changed, "1" as not-changed and "2" as unknown.
    This allows to have a common symobol code for the labels. The symbols can be changed in config.py.

    :param labels: a numpy int array containing the labels for a image pair
    :param conf_section: a ConfigParser Section containing the label specific values

    :return: the input labels refactored as described
    """
    # flag indicating whether a type of label must be changed
    c_set = False
    uc_set = False
    un_set = False

    if int(conf_section.get("changedLabel")) != config.CHANGED_LABEL:
        c_indexes = np.where(labels == int(conf_section.get("changedLabel")))
        c_set = True

    if int(conf_section.get("unchangedLabel")) != config.UNCHANGED_LABEL:
        uc_indexes = np.where(labels == int(conf_section.get("unchangedLabel")))
        uc_set = True

    if conf_section.get("unknownLabel") is not None:
        if int(conf_section.get("unknownLabel")) != config.UNKNOWN_LABEL:
            un_indexes = np.where(labels == int(conf_section.get("unknownLabel")))
            un_set = True

    if c_set:
        labels[c_indexes] = config.CHANGED_LABEL
    if uc_set:
        labels[uc_indexes] = config.UNCHANGED_LABEL
    if un_set:
        labels[un_indexes] = config.UNKNOWN_LABEL

    return labels


def generate_set(img1, img2, labels, keep_unlabeled):
    """
    Function taking two images, the respective labels and returns a 2-dim array containing the pair of pixel
    before/after and a 1-dim array containing the label for the pair.
    The unlabeled pixels (marked as config.UNKNOWN_LABEL) can be removed, if necessary

    :param img1: a 2-dim numpy array of shape (height x width, spectral bands)
    :param img2: a 2-dim numpy array of shape (height x width, spectral bands) - the same shape as img1
    :param labels: a 1-dim numpy array containing a label for each pair
    :param keep_unlabeled: a Boolean flag. If True, keeps all the unlabeled pair of pixels (for testing purposes)

    :return: a 3-dim numpy array of pixel pairs of shape (height x width, 2, spectral bands) and the "labels" array

    """
    pair_list = []
    label_list = []
    for i in range(0, img1.shape[0]):
        if keep_unlabeled or labels[i] != config.UNKNOWN_LABEL:
            pair_list.append([img1[i], img2[i]])
            label_list.append(labels[i])
    return np.asarray(pair_list), np.asarray(label_list)


def generate_pca_feature_set(pairs):
    """
    Function that creates a new feature set containing the concatenated spectral bands for the pixels of first image
    and of second image. Then, PCA (Principal Component Analysis) is applied to the newly created dataset, extracting 2
    principal components.

    :param pairs: pixel pairs of the test set

    :return: feature set to which PCA has been applied
    """

    # generating the new feature set concatenating spectral bands of the two images
    feature_set = np.reshape(pairs, (pairs.shape[0], pairs.shape[1] * pairs.shape[2]))

    # standardizing the feature set
    std_feature_set = StandardScaler().fit_transform(feature_set)

    # applying PCA to the standardized feature set
    pca = PCA(n_components=2)
    pca_feature_set = pd.DataFrame(pca.fit_transform(std_feature_set),
                                   columns=['Principal Component 1', 'Principal Component 2'])

    return pca_feature_set


def load_dataset(name, conf):
    """
    Function loading a dataset containing pairs of satellite multi-spectral or hyper-spectral
    as list of 3-dim numpy arrays of shape (height, width, spectral bands) and the respective pixel-wise labels
    as list of 2-dim numpy array (height, width).
    the dataset must be stored in three different directories (before images, after images and labels) and each
    triplet of files (or directories) must have the same name.
    it also checks whether the dataset is available or not.

    :param name: the name of the dataset to be loaded. If it doesn't exist, an exception is raised
    :param conf: a config parser instance pre-loaded

    :return: four lists containing
        - the first images
        - the second images
        - the labels
        - the names of the triplets (before image, after image, label)
    """
    # checking dataset existence
    if name not in conf.sections():
        raise ValueError(name + " dataset not available")

    # loading paths from settings
    print("Info: LOADING DATASET " + name + "...")
    imgAList = []
    beforepath = conf[name].get("imgAPath")

    imgBList = []
    afterpath = conf[name].get("imgBPath")

    labellist = []
    labelpath = conf[name].get("labelPath")

    i = 0
    namelist = []
    for file in os.listdir(beforepath):
        # for each triplet of images, the "before" image is appended in imgAList, the "after" in imgBList and
        # the label in labellist.
        imgAList.append(load_image(beforepath + os.sep + file, conf[name]))
        imgBList.append(load_image(afterpath + os.sep + file, conf[name]))
        labellist.append(load_label(labelpath + os.sep + file, conf[name]))
        namelist.append(os.path.splitext(file)[0])
        print(str(i + 1) + "/" + str(len(os.listdir(beforepath))) + " pair(s) loaded")
        i = i + 1

    return imgAList, imgBList, labellist, namelist


def load_image(path, conf_section):
    """
    Function loading a single image of a dataset. If the image is in a unsupported format, an exception is raised.

    :param path: string containing the path where the image is stored.
    :param conf_section: a configparser section instance containing the information about the dataset.

    :return: a 3-dim array containing the hyper- or multi-spectral image.
    """
    if ".mat" in path:

        return read_mat(path, conf_section.get("matLabel"))
    elif os.path.isdir(path):
        if ".tif" in os.listdir(path)[0]:
            return read_decompressed_tif(path)

    raise NotImplementedError("Error: CANNOT LOAD NON-MAT FILES")


def read_decompressed_tif(path):
    """
    Function returning an image read from the directory "path".
    since the .tif can be read as a 2-dim array, the image is "decompressed" and stored as n files, each
    one containing the values of a spectral band. In order to work correctly, the
    files must be named in the same alpha-numerical order as the bands are meant to be.

    :param path: string containing the path of the directory where the layers are stored

    :return: a 3-dim numpy array containing the image in pixel-vector format
    """
    files = os.listdir(path)
    files.sort()
    image_cube = None
    for band in files:
        im = PIL.Image.open(path + os.sep + band)
        if image_cube is None:
            image_cube = np.empty((im.size[1], im.size[0], 0))
        image_cube = np.append(image_cube, np.array(im).reshape((im.size[1], im.size[0], 1)), axis=2)
    return image_cube


def load_label(path, conf_section):
    """
    Function loading a single label file of a dataset. If the file is in a unsupported format, an exception is raised.

    :param path: string containing the path where the labels are stored.
    :param conf_section: a configparser section instance containing the information about the dataset.

    :return: a 2-dim array containing the labels for a pair of images
    """
    if ".mat" in path:
        return read_mat(path, conf_section.get("matLabel"))
    if ".tif" in path:
        return np.array(PIL.Image.open(path))
    raise NotImplementedError("Error: CANNOT LOAD LABEL FILE FORMAT")


def preprocessing(limgA, limgB, llabel, conf_section, keep_unlabeled, apply_rescaling=True):
    """
    Function that takes in input a pair of list of images and a pixel-wise label map list and returns
    an array of minmaxscaled (if requested) pixel pairs and an array of refactored labels.
    All the pixel pairs labeled as "unknown" can be discarded.

    :param limgA: a list of 3-dim numpy array of shape (height, width, spectral bands) from where the
                  first pixel of the pair will be taken
    :param limgB: a list of 3-dim numpy array of shape (height, width, spectral bands) from where the
                  second pixel of the pair will be taken
    :param llabel: a list of 2-dim array of shape (height, width) containing the label for each pixel pair.
    :param conf_section: a config parser section instance containing info obout the dataset.
    :param keep_unlabeled: a Boolean flag. If True, keeps all the unlabeled pair of pixels (for testing purposes)
    :param apply_rescaling: a Boolean flag. If True applies the minmax scaling on the pair of pixels (see minmax_pair)

    :return:a list containing:
            - an array containing the labeled pairs of pixels from the images
            - an array containing the labels of the respective pairs
    """

    print("Info: STARTING PREPROCESSING...")
    # loading and linearization of the "A-type" images of the dataset
    imgA = np.empty(shape=(0, limgA[0].shape[2]))
    for img in limgA:
        imgr = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
        imgA = np.append(imgA, imgr, axis=0)

    # loading and linearization of the "B-type" images of the dataset
    imgB = np.empty(shape=(0, limgB[0].shape[2]))
    for img in limgB:
        imgr = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
        imgB = np.append(imgB, imgr, axis=0)

    # min maxing
    if apply_rescaling is True:
        print("Info: APPLYING RESCALING...")
        imgA, imgB = minmax_pair(imgA, imgB)

    # linearization and refactoring of the labels
    label = np.empty(shape=(0))
    for l in llabel:
        lr = np.reshape(l, l.shape[0] * l.shape[1])
        label = np.append(label, refactor_labels(lr, conf_section))

    # pair generation
    print("Info: STARTING PAIRING PROCEDURE...")
    pairs, label = generate_set(imgA, imgB, label, keep_unlabeled)

    return pairs, label
