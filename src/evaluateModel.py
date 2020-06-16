import os

import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, auc

import config.settings as stt
import config.constants as const
import src.dataset as dset
import models.baseModel as base_model

import matplotlib.pyplot as plt


class EvaluateModel:


    def __init__(self):
        self.dataset = dset.Dataset.getInstance()
        self.results = {}


    if const.VERBOSE:
        def print_msg(self, msg):
            """ Prints the given message if VERBOSE is True

                Parameters: msg (str)

                Returns:
                    void
            """
            print(msg)
    else:
        print_msg = lambda msg: None


    def __aggregate_blocks(self, y_pred):

        if const.AGGREGATE_BLOCK_NUM == 1:
            return y_pred
        
        if stt.sel_authentication_type == stt.AuthenticationType.BINARY_CLASSIFICATION:

            pos_pred_tail = int(const.TRAIN_TEST_SPLIT_VALUE / 2)

            y_pred = y_pred.astype(float)
            for i in range(pos_pred_tail - const.AGGREGATE_BLOCK_NUM + 1):
                y_pred[i] = np.average(y_pred[i : i + const.AGGREGATE_BLOCK_NUM], axis=0)

            for i in range(pos_pred_tail, len(y_pred) - const.AGGREGATE_BLOCK_NUM + 1):
                y_pred[i] = np.average(y_pred[i : i + const.AGGREGATE_BLOCK_NUM], axis=0)

        return y_pred


    # Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from predicted scores
    def __get_auc_result(self, model_name, testX, y_true):

        y_pred = base_model.BaseModel.predict_model(model_name, testX)
        y_pred = self.__aggregate_blocks(y_pred[:, 0])
        y_true = np.argmax( y_true, axis=1 )
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=0)

        return auc(fpr, tpr)

    def plot_ROC(self, userid, fpr, tpr, roc_auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='red', linewidth=4,
            lw=lw, label='ROC görbe (terület = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linewidth=4, lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Hamis pozitív arány (FPR)', fontsize=28,)
        plt.ylabel('Igaz pozitív arány (TPR)', fontsize=28,)
        plt.title('ResNet modellel mért ROC görbe user9 felhasználóra', fontsize=30)
        plt.legend(fontsize=28, loc="lower right")
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.show()


    # Computes Accuracy
    def __get_acc_result(self, model_name, testX, y_true):

        y_pred = base_model.BaseModel.predict_model(model_name, testX)
        y_pred = self.__aggregate_blocks(y_pred)
        y_pred = np.argmax( y_pred, axis=1)
        y_true = np.argmax( y_true, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        
        return accuracy


    def __get_confusion_matrix(self, model_name, testX, y_true):

        y_pred = base_model.BaseModel.predict_model(model_name, testX)
        y_pred = self.__aggregate_blocks(y_pred)
        y_pred = np.argmax( y_pred, axis=1)
        y_true = np.argmax( y_true, axis=1)
        conf_matrix = confusion_matrix(y_true, y_pred)

        return conf_matrix


    def __get_evaluation_score(self, arg, model_name, testX, y_true):
        switcher = { 
            1: self.__get_acc_result,
            2: self.__get_auc_result,
            3: self.__get_confusion_matrix
        } 

        func = switcher.get(arg, lambda: "Wrong evaluation metric!")
        return func(model_name, testX, y_true)


    def __evaluate_model_by_metrics(self, user, model_name, testX, y_true):

        evaluation_result = []

        if type(stt.sel_evaluation_metrics) != type([]):
            stt.sel_evaluation_metrics = [stt.sel_evaluation_metrics]
            
        # Evaluating model with all selected metrics
        for metric in stt.sel_evaluation_metrics:
            self.print_msg('\n' + str(metric) + ' value:')
            value = self.__get_evaluation_score(metric.value, model_name, testX, y_true)
            # Saving result to a map for further use
            evaluation_result.append(value)
            self.print_msg(value)

        self.results[user] = evaluation_result


    def __evaluate_model_by_method(self, user, model_name):

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            testX, y_true = self.dataset.create_test_dataset_for_authentication(user)
        else:
            testX, y_true = self.dataset.create_test_dataset_for_identification()
        y_true = to_categorical(y_true)

        # Reshapes data for TIME_DISTRIBUTED model input
        if stt.sel_model == stt.Model.TIME_DISTRIBUTED:
            n_steps, n_length = 4, int(testX.shape[1] / 4)
            testX = testX.reshape((testX.shape[0], n_steps, n_length, testX.shape[2]))

        self.print_msg('\nTest dataset shape: ')
        self.print_msg(testX.shape)
        self.print_msg(y_true.shape)
        
        self.__evaluate_model_by_metrics(user, 'best_' + model_name, testX, y_true)


    def __evaluate_model_action_based_single_user(self, model_name):
        self.print_msg('\nEvaluating model for user: ' + const.USER_NAME + '...')
        model_name = const.USER_NAME + '_' + model_name
        self.__evaluate_model_by_method(const.USER_NAME, model_name)
        self.print_msg('\nEvaluating model finished.\n')


    def __evaluate_model_action_based_all_user(self, model_name):
        userArr = stt.get_users()

        for user in userArr:
            self.print_msg('\nEvaluatig model for user: ' + user + '...')
            tmp_model_name = user + '_' + model_name
            self.__evaluate_model_by_method(user, tmp_model_name)
            self.print_msg('\nEvaluatig model finished.\n')
            
            
    def __action_based_evaluation(self, model_name):

        if stt.sel_evaluate_user_number == stt.EvaluateUserNumber.SINGLE:
            self.__evaluate_model_action_based_single_user(model_name)
        else:
            self.__evaluate_model_action_based_all_user(model_name)


    def __evaluate_model_for_identification(self, model_name):
        model_name = 'identification_' + model_name
        self.__evaluate_model_by_method(None, model_name)


    def evaluate_model(self):
        self.results = {}
        
        model_name = str(stt.sel_model) + '_' + str(stt.sel_dataset) + '_' + str(const.BLOCK_SIZE) + '_' + str(stt.BLOCK_NUM) + '_trained.hdf5'

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:

            if stt.sel_evaluation_type == stt.EvaluationType.ACTION_BASED:
                self.__action_based_evaluation(model_name)
            else:
                # It has to implement session based evaluation
                # TODO
                raise NotImplementedError

            file_name = 'authentication_' + model_name
        else:
            self.__evaluate_model_for_identification(model_name)
            file_name = 'identification_' + model_name
            
        if stt.print_evaluation_results_to_file:    
            self.__print_results_to_file(self.results, file_name)


    def __print_results_to_file(self, results, file_name):
        """ Prints the given result to file

            Parameters:
                map - contains all result in form: (metric name, value)
                str - filename for saving results

            Returns:
                None
        """
        if not os.path.exists(const.RESULTS_PATH):
            os.makedirs(const.RESULTS_PATH)

        file_name = file_name[:len(file_name) - 13] + '_' + stt.sel_raw_feature_type.value
        file = open(const.RESULTS_PATH + '/' + file_name + '.csv', 'w')
        file.write('username,')

        for metric in stt.sel_evaluation_metrics:
            file.write(str(metric) + ',')

        file.write('\n')
        
        for user, values in results.items():
            file.write(str(user) + ',')
            
            for value in values:

                if type(value) is np.ndarray:
                    value = str(value.tolist())
                    value = value.replace(',', ' ')
                
                file.write(str(value) + ',')

            file.write('\n')

        file.close()


if __name__ == "__main__":
    em = EvaluateModel()