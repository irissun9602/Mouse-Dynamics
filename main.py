# Setting random states to get reproducible results

#----------------------------- Keras reproducible CPU ------------------#
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np
import random as rn

sd = 1 # Here sd means seed.
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED']=str(sd)

from keras import backend as K
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#-----------------------------------------------------------------------#

#----------------------------- Keras reproducible GPU ------------------#

#from numpy.random import seed
#seed( 42 )
#from tensorflow import set_random_seed
#set_random_seed( 42 )

#------------------------------------------------------------------------#

import config.settings as stt
import config.constants as const
import src.trainModel as trainModel
import src.evaluateModel as evaluateModel


def main():

    if stt.sel_method == stt.Method.TRAIN or stt.sel_method == stt.Method.TRANSFER_LEARNING:
        tm = trainModel.TrainModel()
        tm.train_model()
            
    if stt.sel_method == stt.Method.EVALUATE:
        em = evaluateModel.EvaluateModel()  
        em.evaluate_model()


if __name__ == "__main__":

    # Performing only one action (train or test)
    if stt.sel_settings_source == stt.SettingsSource.FROM_PY_FILE:
        main()

    # Performing multiple actions (train or test)
    if stt.sel_settings_source == stt.SettingsSource.FROM_XML_FILE:
        import config.parser as pars
    
        parser = pars.Parser.getInstance()
        
        while parser.has_next_action():
            parser.execute_next_action()
            main()