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
    if parser["settings"].getboolean("training") is True:

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

    # TESTING ROUTINE
    else:
        # declaring placeholder to be printed when no fine tuning is executed
        pseudo_qty = "no fine tuning"
        extraction_time = "-"
        fine_time = "-"
        loss = "-"
        val_loss = "-"
        val_acc = "-"
        epochs = "-"
        pseudo_accuracy = "-"
        pseudo_accuracy_corr = "-"

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
            ft_conf = int(parser["settings"].get("fine_tuning"))

            if ft_conf >= 0:
                # loading the dictionary containing distances, threshold and shape
                pseudo_file = open(parser[test_set].get("pseudoPath") + os.sep + names[j] + ".pickle", "rb")
                pseudo_dict = pickle.load(pseudo_file)
                pseudo_file.close()

                # generating spatial corrected pseudo_labels for metrics computing
                pseudo_truth = np.where(pseudo_dict["distances"] > pseudo_dict["threshold"], config.CHANGED_LABEL,
                                        config.UNCHANGED_LABEL)
                pseudo_truth = pu.spatial_correction(np.reshape(pseudo_truth, pseudo_dict["shape"])).ravel()

                if ft_conf == 0:
                    # selecting all the pseudo-labels

                    pseudo_qty = "all"
                    toc_extraction = 0
                    tic_extraction = 0
                    data_used = np.arange(0, len(pseudo_dict["distances"]))
                    labels_used = pseudo_truth

                elif ft_conf == 1:
                    # selecting the "percentage"*100% best pseudo-labels

                    percentage = float(parser["settings"].get("pseudo_percentage"))
                    pseudo_qty = str(percentage * 100) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_by_percentage_sam(pseudo_dict, percentage)
                    toc_extraction = time.time()

                elif ft_conf == 2:
                    # selecting the best pseudo-labels by neighbourhood with "radius" radius

                    radius = int(parser["settings"].get("pseudo_radius"))
                    pseudo_qty = "r=" + str(radius)
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_by_neighborhood(pseudo_dict, radius)
                    toc_extraction = time.time()

                elif ft_conf == 3:
                    # selecting the "percentage"*100% best pseudo-labels and using real labels for remaining data

                    percentage = float(parser["settings"].get("pseudo_percentage"))
                    pseudo_qty = str(percentage * 100) + "%"
                    real_qty = str(round(1 - percentage, 1) * 100) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_plus_labels_by_percentage_sam(pseudo_dict, percentage, img_label)
                    toc_extraction = time.time()

                elif ft_conf == 4:
                    # selecting only real labels discarded from the extraction of pseudo labels by percentage

                    percentage = float(parser["settings"].get("pseudo_percentage"))
                    pseudo_qty = str(percentage * 100) + "%"
                    real_qty = str(round(1 - percentage, 1) * 100) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_percentage_sam(pseudo_dict, percentage, img_label)
                    toc_extraction = time.time()

                elif ft_conf == 5:
                    # selecting the best pseudo-labels by neighbourhood with "radius" radius and using real labels for
                    # remaining data

                    radius = int(parser["settings"].get("pseudo_radius"))
                    pseudo_qty = "r=" + str(radius)
                    tic_extraction = time.time()
                    data_used, labels_used = pu.pseudo_plus_labels_by_neighborhood(pseudo_dict, img_label, radius)
                    toc_extraction = time.time()

                elif ft_conf == 6:
                    # selecting only real labels discarded from the extraction of pseudo labels by neighborhood

                    radius = int(parser["settings"].get("pseudo_radius"))
                    pseudo_qty = "r=" + str(radius)
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_neighborhood(pseudo_dict, img_label, radius)
                    toc_extraction = time.time()

                elif ft_conf == 7:
                    # uncertainty sampling (active learning)

                    percentage = float(parser["settings"].get("uncertainty_percentage"))
                    real_qty = str(percentage * 100) + "%"
                    tic_extraction = time.time()
                    data_used, labels_used = pu.labels_by_percentage_siamese(model, pairs, img_label, percentage)
                    toc_extraction = time.time()

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
            if ft_conf <= 2:
                plt.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo" + "_heatmap.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 3:
                plt.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo_" + real_qty + "_real" + "_heatmap.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 4:
                plt.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                            + "_real" + "_heatmap.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 5:
                plt.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo" + "_real" + "_heatmap.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 6:
                plt.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_real" + "_heatmap.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 7:
                plt.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                            + "_uncertainty" + "_heatmap.png", dpi=300, bbox_inches='tight')

            # 1. confusion matrix
            cm = skm.confusion_matrix(img_label, prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

            # 2. getting the metrics
            metrics = s.get_metrics(cm)

            # 3. Opening a new file
            print("Info: SAVING THE " + str(j + 1) + "° RESULT")
            if ft_conf <= 2:
                file = open(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo" + ".csv", "w")
            elif ft_conf == 3:
                file = open(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo_" + real_qty + "_real" + ".csv", "w")
            elif ft_conf == 4:
                file = open(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                            + "_real" + ".csv", "w")
            elif ft_conf == 5:
                file = open(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo" + "_real" + ".csv", "w")
            elif ft_conf == 6:
                file = open(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_real" + ".csv", "w")
            elif ft_conf == 7:
                file = open(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                            + "_uncertainty" + ".csv", "w")

            # 4. printing column names, number of examples and the used threshold
            file.write("total_examples, threshold")

            for k in metrics.keys():
                file.write(", " + k)

            file.write(", prediction_time")

            for k in metrics.keys():
                file.write(", " + k + "_correction")

            file.write(", correction_time, pseudo_qty, extraction_time," +
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
            if ft_conf <= 2:
                fig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo" + ".png", dpi=300, bbox_inches='tight')
            elif ft_conf == 3:
                fig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo_" + real_qty + "_real" + ".png", dpi=300, bbox_inches='tight')
            elif ft_conf == 4:
                fig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                            + "_real" + ".png", dpi=300, bbox_inches='tight')
            elif ft_conf == 5:
                fig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_pseudo" + "_real" + ".png", dpi=300, bbox_inches='tight')
            elif ft_conf == 6:
                fig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                            + "_real" + ".png", dpi=300, bbox_inches='tight')
            elif ft_conf == 7:
                fig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                            + "_uncertainty" + ".png", dpi=300, bbox_inches='tight')

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
            file.write(", " + str(correction_time) + ", " + pseudo_qty + ", " + str(extraction_time) +
                       ", " + str(epochs) + ", " + str(fine_time) + ", " + str(loss) + ", " + str(val_loss) +
                       ", " + str(val_acc) + ", " + str(pseudo_accuracy) + ", " + str(pseudo_accuracy_corr))
            file.write("\n")
            file.close()

            scfig = pu.plot_maps(corrected_map, ground_t)
            if ft_conf <= 2:
                scfig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                              + "_pseudo" + "_corrected.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 3:
                scfig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                              + "_pseudo_" + real_qty + "_real" + "_corrected.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 4:
                scfig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                              + "_real" + "_corrected.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 5:
                scfig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                              + "_pseudo" + "_real" + "_corrected.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 6:
                scfig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + pseudo_qty
                              + "_real" + "_corrected.png", dpi=300, bbox_inches='tight')
            elif ft_conf == 7:
                scfig.savefig(config.STAT_PATH + test_set + "_" + names[j] + "_on_" + model_name + "_" + real_qty
                              + "_uncertainty" + "_corrected.png", dpi=300, bbox_inches='tight')

            i = i + lab.size
            j += 1
