# imports
from numpy import pi
from os import sep

"""
    Paths --------------------------------------------------------------------------------------------------------------
"""
# Path for the file containing the dataset configurations and the settings for the scripts
DATA_CONFIG_PATH = "net.conf"

# Path where the model are saved or stored
MODEL_SAVE_PATH = "model" + sep

# Path where the file containing stats are saved
STAT_PATH = "stat" + sep


"""
    Constants ----------------------------------------------------------------------------------------------------------
"""
# Label for the changed pixels
CHANGED_LABEL = 0

# Label for the unchanged pixels
UNCHANGED_LABEL = 1

# Label for the unknown pixels
UNKNOWN_LABEL = 2

# Dictionary containing the available fixed threshold used for training accuracy during training.
# A threshold must be added for each distance function implemented.
AVAILABLE_THRESHOLD = {'euclidean_dist': 0.5, 'SAM': pi/4}

# Dictionary containing the available margin used for the contrastive loss function.
# A threshold must be added for each distance function implemented.
AVAILABLE_MARGIN = {'euclidean_dist': 1, 'SAM': pi/2}

# Constant containing the actual value used as threshold for training accuracy during training.
PRED_THRESHOLD = 0.5

# Constant containing the actual value used as margin for the contrastive loss function.
MARGIN = 1

# Constant containing the percentage of the training set to be used as validation set
VAL_SPLIT = 0.2

# Constant containing the number of evaluations to be performed during hyperparameter optimization
MAX_EVALS = 30

# List containing the color map to be used in the prediction/pseudo-labels/ground truth plots
# the colors are assigned in ascending order
# r = 0 => changed, b = 1 => unchanged, y = 2 => unknown
COLOR_MAP = ['r', 'b', 'y']


"""
    Global variables used for model optimization purposes (hyperas) ----------------------------------------------------
"""
# List containing the confusion matrices computed during each iteration on the test set
test_cm = None

# List containing the confusion matrices computed during each iteration on the validation set
val_cm = None

# Float used to store the current best score (validation loss) during Hyperas iterations
best_score = float("inf")

# Variable storing a reference to the current best model during Hyperas iterations
best_model = None

# Float used to store the time elapsed in the current best run
best_time = 0

# Variables used to pass the train and test set to the optimization function
train_set = None
test_set = None
train_labels = None
test_labels = None

# Variable storing the selected distance function
selected_distance = None

# Variable storing the list of values among which the optimization function will select the batch size
batch_size = None

max_dropout = 0

# Variable storing the list of values among which the optimization function will select the number of neurons of
# the first layer
neurons = None

# Variable storing the list of values among which the optimization function will select the number of neurons of
# the second layer
neurons_1 = None

# Variable storing the list of values among which the optimization function will select the number of neurons of
# the third layer
neurons_2 = None

# Boolean indicating whether to add or not the fourth layer (see build_net)
fourth_layer = False
