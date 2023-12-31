import config.settings as stt


""" 
    Global constant values
    ##################################
"""

# Defines the samples number in one continuous mouse movement.
BLOCK_SIZE = 128


# Evaluate model using AGGREGATE_BLOCK_NUM number of blocks.
AGGREGATE_BLOCK_NUM = 1


# Defines the username for
# TRAIN and EVALUATE model
USER_NAME = 'user9'


# Defines the model name for initializing weights. 
# Used only for transfer learning
USED_MODEL_FOR_TRANSFER_LEARNING = 'best_User11_Model.CNN_Dataset.DFL_128_300_trained.hdf5'


# It is relevant in OCC measurements.
# If sel_occ_features is FEATURES_FROM_CNN:
# This model serves as feature extractor.
USED_MODEL_FOR_OCC_FEATURE_EXTRACTION = 'dfl_2000_ResNet.hdf5'


# Defines train-test split ratio.
# Only needs if TRAIN_TEST_SPLIT_TYPE is TRAIN_AVAILABLE
# If its value is between (0, 1) then represents the proportion of the dataset to include in the train split.
# If int, represents the absolute number of train samples.
TRAIN_TEST_SPLIT_VALUE = 0.8


# Defines the batch size for model training.
BATCH_SIZE = 32


# Trained models path.
TRAINED_MODELS_PATH = 'C:/Anaconda projects/Software_mod/trainedModels'


# Evaluation results path.
RESULTS_PATH = 'C:/Anaconda projects/Software_mod/evaluationResults'


# Saved images path.
SAVED_IMAGES_PATH = 'C:/Anaconda projects/Software_mod/savedImages'


# Config file location for the automated sript.
# If sel_settings_source is FROM_XML_FILE, then the software runs model training/testing
# In an automated way, defined in config.xml file.
CONFIG_XML_FILE_LOCATION = './config/config.xml'


# Defines random state to initialize environment variables.
RANDOM_STATE = 42


# Set verbose mode on/off.
VERBOSE = True


# Maximum screen sizes in pixels for saturating outlayer values.
MAX_WIDTH = 4000
MAX_HEIGHT = 4000


""" 
    Specific constants for each dataset
    ###################################
"""
STATELESS_TIME = 2

if stt.sel_dataset == stt.Dataset.BALABIT:
    # Defines the interval when no user interaction occurred.
    # It is measured in seconds.
    STATELESS_TIME = 2

    # Test files path.
    TEST_FILES_PATH = r'C:\Users\user\Downloads\Mouse-Dynamics-Challenge-master\Mouse-Dynamics-Challenge-master\test_files'

    # Test labels path.
    TEST_LABELS_PATH = r'C:\Users\user\Downloads\Mouse-Dynamics-Challenge-master\Mouse-Dynamics-Challenge-master\public_labels.csv'

    # Training files path.
    TRAIN_FILES_PATH = r'C:\Users\user\Downloads\Mouse-Dynamics-Challenge-master\Mouse-Dynamics-Challenge-master\training_files'


if stt.sel_dataset == stt.Dataset.DFL:
    TEST_FILES_PATH = r'C:\Users\user\Downloads\DFL'
    TRAIN_FILES_PATH = r'C:\Users\user\Downloads\DFL'
    STATELESS_TIME = STATELESS_TIME * 1000


def setter(property_name, arg):
    """ Function for setting these values:
        It sets property_name with arg value.

        Parameters:
            property_name (str): Enum type
            arg (str): Enum type value

        Returns:
            None
    """ 
    
    if arg[0].isdigit():
        value = int(arg[0])
    else:
        if arg[0] == 'True' or arg[0] == 'False':
            value = eval(arg[0])
        else:
            value = arg[0]

    globals()[property_name] = value