import settings as stt
import constants as const

import os
from keras.models import load_model


class BaseModel:


    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.is_trained = False

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            self.n_output = 2
        
        if stt.sel_user_recognition_type == stt.UserRecognitionType.IDENTIFICATION:
            self.n_output = len( stt.get_users() )


    def create_model(self):
        raise NotImplementedError


    def __set_weights_from_pretrained_model(self, model_path):

        try:
            old_model = load_model(model_path)
        except:
            return
        print('itt----------------------------------------------------------------------------')

        for i in range(len(old_model.layers) - 1):
            self.model.layers[i].set_weights(old_model.layers[i].get_weights())


    def train_model(self):
        self.is_trained = True

        if stt.sel_user_recognition_type == stt.UserRecognitionType.IDENTIFICATION and stt.enable_transfer_learning:
            self.__set_weights_from_pretrained_model(const.TRAINED_MODELS_PATH + '/' + self.model_name)
    
    
    def get_trained_model(self):

        if self.is_trained:
            return self.model
        return False


    def save_model(self):

        # Save the model
        if not os.path.exists(const.TRAINED_MODELS_PATH):
            os.makedirs(const.TRAINED_MODELS_PATH)

        self.model.save(const.TRAINED_MODELS_PATH + '/' + self.model_name)


    @staticmethod
    def predict_with_pretrained_model(model_name, x_data):
        model_path = const.TRAINED_MODELS_PATH + '/' + model_name
        model = load_model(model_path)

        if stt.sel_model == stt.Model.TIME_DISTRIBUTED:
            n_steps, n_length = 4, int(const.BLOCK_SIZE / 4)
            x_data = x_data.reshape((x_data.shape[0], n_steps, n_length, 2))

        return model.predict(x_data)