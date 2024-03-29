# full imports
import time
import config
import configparser
import pickle
import os

# aliases
import dataprocessing as dp
import siamese as s
import predutils as pu
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt

# single imports
from skimage.filters import threshold_otsu
from kneed import KneeLocator
from sklearn.cluster import KMeans

np.random.seed(43)

"""
    Main script for training and testing a Siamese Net Model.
        - If "training"==True in net.conf, the training routine is executed.
          The unlabeled pairs are automatically removed from the specified training set.
          The training routine executes "hyperparam_search" on the specified test and training set. 
          The number of evaluations can be changed, as every other "internal" constant, in config.py
        - If "training"==False in net.conf, the testing routine is executed.
          The unlabeled pairs are kept in the specified test set, so that the prediction can be performed on the whole 
          images. If specified, fine tuning is executed for each image in the test set, before executing the prediction.
        - This script also saves the .csv file with the result of the testing and the map plots, before and after
          the spatial correction.
"""

if __name__ == '__main__':
    # opening the parser
    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    # reading the names of train and test set and of the model to be learned/loaded
    train_set = parser["settings"].get("train_set")
    test_set = parser["settings"].get("test_set")
    model_name = parser["settings"].get("model_name")

    print("Selected train dataset: " + train_set)
    print("Selected test dataset: " + test_set)
    print("Selected model: " + model_name)

    # TRAINING ROUTINE
    if parser["settings"].getint("operation") == 0:

        # reading the distance function and settings for the learning phase
        if parser["settings"].get("distance") == "ED":

            hyperas_sett = "hyperas settings ED"
            distance_func = s.euclidean_dist
        elif parser["settings"].get("distance") == "SAM":

            hyperas_sett = "hyperas settings SAM"
            distance_func = s.SAM
        else:
            raise NotImplementedError(
                "Error: DISTANCE FUNCTION " + parser["settings"].get("distance") + " NOT IMPLEMENTED")

        # loading the pairs
        train_a_img, train_b_img, train_labels, train_names = dp.load_dataset(train_set, parser)
        test_a_img, test_b_img, test_labels, test_names = dp.load_dataset(test_set, parser)

        # executing preprocessing
        x_train, y_train = dp.preprocessing(train_a_img, train_b_img, train_labels, parser[train_set],
                                            keep_unlabeled=False,
                                            apply_rescaling=parser["settings"].getboolean("apply_rescaling"))
        x_test, y_test = dp.preprocessing(test_a_img, test_b_img, test_labels, parser[test_set],
                                          keep_unlabeled=True,
                                          apply_rescaling=parser["settings"].getboolean("apply_rescaling"))

        # executing hyperparameters automatic search
        print("Info: STARTING HYPERSEARCH PROCEDURE")
        model, run = s.hyperparam_search(x_train, y_train, x_test, y_test, distance_func, model_name,
                                         parser[hyperas_sett])

    # CLUSTERING ROUTINE
    elif parser["settings"].getint("operation") == 1:

        # dataset loading
        first_img, second_img, labels, names = dp.load_dataset(test_set, parser)
        x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[test_set],
                                          keep_unlabeled=True,
                                          apply_rescaling=parser["settings"].getboolean("apply_rescaling"))

        # clustering
        i = 0
        j = 0
        for lab in labels:
            pairs = x_test[i:i + lab.size, :]
            pca_dataset = dp.generate_pca_dataset(pairs)
            cluster_path = parser[test_set].get("clusterPath") + os.sep + names[j] + ".pickle"

            # calculating best number of clusters
            print("Info: CALCULATING BEST NUMBER OF CLUSTERS...")
            sse = []
            for k in range(50, 76):
                kmeans = KMeans(n_clusters=k, random_state=43)
                kmeans.fit(pca_dataset)
                sse.append(kmeans.inertia_)
            kl = KneeLocator(range(50, 76), sse, curve="convex", direction="decreasing")
            k = kl.elbow

            # initializing and fitting the K-Means model
            kmeans = KMeans(n_clusters=k, random_state=43)
            kmeans.fit(pca_dataset)

            # creating a dictionary of pixels for each cluster and serializing it to a file
            clusters = np.array(kmeans.predict(pca_dataset))
            clusters_dict = {}
            for clu in range(k):
                clusters_dict[clu] = np.where(clusters == clu)[0]
            cluster_file = open(cluster_path, "wb")
            pickle.dump(clusters_dict, cluster_file)
            cluster_file.close()

            i = i + lab.size
            j += 1

    # TESTING ROUTINE
    elif parser["settings"].getint("operation") == 2:

        # declaring placeholder to be printed when no fine tuning is executed
        pseudo_qty = "no pseudo labels"
        real_qty = "no real labels"
        extraction_time = "-"
        fine_time = "-"
        loss = "-"
        val_loss = "-"
        val_acc = "-"
        epochs = "-"
        pseudo_accuracy = "-"
        pseudo_accuracy_corr = "-"
        type_execution = "no ft"

        # dataset loading
        first_img, second_img, labels, names = dp.load_dataset(test_set, parser)
        x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[test_set],
                                          keep_unlabeled=True,
                                          apply_rescaling=parser["settings"].getboolean("apply_rescaling"))

        # parameters loading and model building
        print("Info: LOADING THE MODEL...")
        param_file = open(config.MODEL_SAVE_PATH + model_name + "_param.pickle", "rb")
        parameters = pickle.load(param_file)
        param_file.close()
        model = s.build_net(x_test[0, 0].shape, parameters)

        i = 0
        j = 0
        # The model will be tuned fresh for each image
        # and each image must be extracted from the preprocessed matrix
        for lab in labels:
            pairs = x_test[i:i + lab.size, :]
            img_label = y_test[i:i + lab.size].astype(int)

            # weights loading
            model.load_weights(config.MODEL_SAVE_PATH + model_name + ".h5")

            # Fine tuning phase
            ft = int(parser["settings"].get("fine_tuning"))

            if ft >= 0:
                # loading the dictionary containing distances, threshold and shape
                pseudo_file = open(parser[test_set].get("pseudoPath") + os.sep + names[j] + ".pickle", "rb")
                pseudo_dict = pickle.load(pseudo_file)
                pseudo_file.close()

                # generating spatial corrected pseudo_labels for metrics computing
                pseudo_truth = np.where(pseudo_dict["distances"] > pseudo_dict["threshold"], config.CHANGED_LABEL,
                                        config.UNCHANGED_LABEL)
                pseudo_truth = pu.spatial_correction(np.reshape(pseudo_truth, pseudo_dict["shape"])).ravel()

                if ft == 0:
                    # selecting all the pseudo-labels

                    pseudo_qty = "all"
                    toc_extraction = 0
                    tic_extraction = 0
                    data_used = np.arange(0, len(pseudo_dict["distances"]))
                    labels_used = pseudo_truth
                    type_execution = pseudo_qty + "_pseudo"

                elif ft == 1:
                    # selecting the "percentage"*100% best pseudo-labels

                    percentage = float(parser["settings"].get("pseudo_percentage"))
                    pseudo_qty = str(percentage * 100) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_by_percentage_sam(pseudo_dict, percentage)
                    toc_extraction = time.time()
                    type_execution = pseudo_qty + "_pseudo"

                elif ft == 2:
                    # selecting the best pseudo-labels by neighbourhood with "radius" radius

                    radius = int(parser["settings"].get("pseudo_radius"))
                    pseudo_qty = "r=" + str(radius)
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_by_neighborhood(pseudo_dict, radius)
                    toc_extraction = time.time()
                    type_execution = pseudo_qty + "_pseudo"

                elif ft == 3:
                    # selecting the "percentage"*100% best pseudo-labels and using real labels for remaining data

                    percentage = float(parser["settings"].get("pseudo_percentage"))
                    pseudo_qty = str(percentage * 100) + "%"
                    real_qty = str(round((1 - percentage) * 100, 1)) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_plus_labels_by_percentage_sam(pseudo_dict, percentage, img_label)
                    toc_extraction = time.time()
                    type_execution = pseudo_qty + "_pseudo_" + real_qty + "_real"

                elif ft == 4:
                    # selecting only real labels discarded from the extraction of pseudo labels by percentage

                    percentage = float(parser["settings"].get("pseudo_percentage"))
                    real_qty = str(round((1 - percentage) * 100, 1)) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_percentage_sam(pseudo_dict, percentage, img_label)
                    toc_extraction = time.time()
                    type_execution = real_qty + "_real"

                elif ft == 5:
                    # selecting the best pseudo-labels by neighbourhood with "radius" radius and using real labels for
                    # remaining data

                    radius = int(parser["settings"].get("pseudo_radius"))
                    pseudo_qty = "r=" + str(radius)
                    real_qty = "r=" + str(radius)
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_plus_labels_by_neighborhood(pseudo_dict, img_label, radius)
                    toc_extraction = time.time()
                    type_execution = pseudo_qty + "_pseudo_real"

                elif ft == 6:
                    # selecting only real labels discarded from the extraction of pseudo labels by neighborhood

                    radius = int(parser["settings"].get("pseudo_radius"))
                    real_qty = "r=" + str(radius)
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_neighborhood(pseudo_dict, img_label, radius)
                    toc_extraction = time.time()
                    type_execution = real_qty + "_real"

                elif ft == 7:
                    # uncertainty sampling (active learning)

                    percentage = float(parser["settings"].get("uncertainty_percentage"))
                    real_qty = str(percentage * 100) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_percentage_uncertainty(model, pairs, img_label, percentage)
                    toc_extraction = time.time()
                    type_execution = real_qty + "_uncertainty"

                elif ft == 8:
                    # k-means, randomly selecting examples for each cluster (active learning)

                    percentage = float(parser["settings"].get("kmeans_percentage"))
                    real_qty = str(percentage * 100) + "%"
                    cluster_path = parser[test_set].get("clusterPath") + os.sep + names[j] + ".pickle"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_percentage_k_means_random(img_label, percentage, cluster_path)
                    toc_extraction = time.time()
                    type_execution = real_qty + "_kmeans_random"

                elif ft == 9:
                    # k-means, selecting top uncertain examples for each cluster (active learning)

                    percentage = float(parser["settings"].get("kmeans_percentage"))
                    real_qty = str(percentage * 100) + "%"
                    cluster_path = parser[test_set].get("clusterPath") + os.sep + names[j] + ".pickle"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_percentage_k_means_uncertainty(img_label, percentage,
                                                                                         cluster_path, model, pairs)
                    toc_extraction = time.time()
                    type_execution = real_qty + "_kmeans_uncertainty"

                else:
                    raise ValueError("Error: FINE TUNING CHOICE " + parser["settings"].get("fine_tuning")
                                     + " NOT IMPLEMENTED")

                # computing the extraction time of the pseudo-labels
                extraction_time = toc_extraction - tic_extraction

                # shuffle delle coppie di pixel e delle labels
                taken_samples = np.c_[data_used, labels_used]
                np.random.shuffle(taken_samples)
                data_used = taken_samples[:, 0].astype(int)
                labels_used = taken_samples[:, 1].astype(int)

                # performing fine tuning
                print("Info: PERFORMING FINE TUNING...")
                model, loss, val_loss, val_acc, epochs, fine_time = s.fine_tuning(model,
                                                                                  parameters['batch_size'],
                                                                                  pairs[data_used], labels_used)

                # saving new model
                if ft == 0:
                    model.save(config.MODEL_SAVE_PATH + os.sep + names[j] + "_on_" + model_name +
                               "_FT" + str(ft) + ".h5")
                elif ft == 1 or ft == 2 or ft == 5:
                    model.save(config.MODEL_SAVE_PATH + os.sep + names[j] + "_on_" + model_name +
                               "_FT" + str(ft) + "_" + pseudo_qty + ".h5")
                elif ft == 3:
                    model.save(config.MODEL_SAVE_PATH + os.sep + names[j] + "_on_" + model_name +
                               "_FT" + str(ft) + "_" + pseudo_qty + "_" + real_qty + ".h5")
                else:
                    model.save(config.MODEL_SAVE_PATH + os.sep + names[j] + "_on_" + model_name +
                               "_FT" + str(ft) + "_" + real_qty + ".h5")

            # performing prediction and computing the elapsed time
            print("Info: EXECUTING PREDICTION OF " + names[j] + " " + str(j + 1) + "/" + str(len(labels)))
            tic_prediction = time.time()
            distances = model.predict([pairs[:, 0], pairs[:, 1]])
            toc_prediction = time.time()
            prediction_time = toc_prediction - tic_prediction

            # computing threshold and turning distances into labels
            threshold = threshold_otsu(distances)
            prediction = np.where(distances.ravel() > threshold,
                                  config.CHANGED_LABEL, config.UNCHANGED_LABEL)

            print("Info: COMPUTING THE METRICS...")
            # print the heatmap0
            im = plt.imshow(distances.reshape(lab.shape), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + type_execution +
                        "_heatmap.png", dpi=300, bbox_inches='tight')

            # 1. confusion matrix
            cm = skm.confusion_matrix(img_label, prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

            # 2. getting the metrics
            metrics = s.get_metrics(cm)

            # 3. Opening a new file
            print("Info: SAVING THE " + str(j + 1) + "° RESULT")
            file = open(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + type_execution +
                        ".csv", "w")

            # 4. printing column names, number of examples and the used threshold
            file.write("total_examples, threshold")

            for k in metrics.keys():
                file.write(", " + k)

            file.write(", prediction_time")

            for k in metrics.keys():
                file.write(", " + k + "_correction")

            file.write(", correction_time, pseudo_qty, real_qty, extraction_time," +
                       " ft_epochs, ft_time, ft_loss, ft_val_loss, ft_val_acc, pseudo_acc, pseudo_acc_corrected")

            file.write("\n %d, %f" % (len(img_label), threshold))

            # 5. printing metrics without correction
            for k in metrics.keys():
                file.write(", " + str(metrics[k]))
            file.write(", " + str(prediction_time))

            # 6. saving the map plot
            # a. the prediction is reshaped as a 2-dim array
            lmap = np.reshape(prediction, lab.shape)
            # b. label is refactored singularly in order to provide coherent ground truth
            ground_t = dp.refactor_labels(lab, parser[test_set])
            # c. the maps are plotted with the appropriate function
            fig = pu.plot_maps(lmap, ground_t)
            fig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + type_execution +
                        ".png", dpi=300, bbox_inches='tight')

            print("Info: EXECUTING SPATIAL CORRECTION...")
            # replying steps 1, 2, 3, 5 and 6 after the spatial correction
            # the elapsed time during correction is also recorded
            tic_correction = time.time()
            corrected_map = pu.spatial_correction(lmap)
            toc_correction = time.time()
            correction_time = toc_correction - tic_correction

            print("Info: GETTING AND SAVING THE METRICS AFTER SC...")
            sccm = skm.confusion_matrix(img_label, corrected_map.ravel(),
                                        labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

            scmetrics = s.get_metrics(sccm)

            # if fine tuning is enabled, accuracy with respect to the pseudo labels is computed and printed on file
            if int(parser["settings"].get("fine_tuning")) >= 0:
                pseudocm = skm.confusion_matrix(pseudo_truth, prediction,
                                                labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
                pseudocm_corrected = skm.confusion_matrix(pseudo_truth, corrected_map.ravel(),
                                                          labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
                pseudo_accuracy = s.get_metrics(pseudocm)["overall_accuracy"]
                pseudo_accuracy_corr = s.get_metrics(pseudocm_corrected)["overall_accuracy"]

            for k in scmetrics.keys():
                file.write(", " + str(scmetrics[k]))
            file.write(", " + str(correction_time) + ", " + pseudo_qty + ", " + real_qty + ", " + str(extraction_time) +
                       ", " + str(epochs) + ", " + str(fine_time) + ", " + str(loss) + ", " + str(val_loss) +
                       ", " + str(val_acc) + ", " + str(pseudo_accuracy) + ", " + str(pseudo_accuracy_corr))
            file.write("\n")
            file.close()

            scfig = pu.plot_maps(corrected_map, ground_t)
            scfig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + type_execution +
                          "_corrected.png", dpi=300, bbox_inches='tight')

            i = i + lab.size
            j += 1

    else:
        raise ValueError("Error: OPERATION CHOICE " + parser["settings"].get("operation") + " NOT IMPLEMENTED")
