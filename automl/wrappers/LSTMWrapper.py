from .BaseWrapper import BaseWrapper
from tqdm import tqdm
import copy
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import train_test_split


class LSTMWrapper(BaseWrapper):
    def __init__(self, automl_instance):
        super().__init__(automl_instance)

    def transform_data(self, data):
        self.data = self.automl._data_shift.transform(data)
        self.past_labels = self.automl._data_shift.past_labels
        self.past_lags = self.automl._data_shift.past_lags
        self.oldest_lag = int(max(self.past_lags)) + 1
        self.index_label = self.automl.index_label
        self.target_label = self.automl.target_label
        self.last_x = data.drop(
            [self.index_label, self.target_label], axis=1).tail(1)

        X = self.data[self.past_labels]
        y = self.data[self.target_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.automl.train_val_split, shuffle=False)

        X_train = np.reshape(
            X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(
            X_test.values, (X_test.shape[0], X_test.shape[1], 1))

        self.training = (X_train, y_train.values)
        self.validation = (X_test, y_test.values)

    def create_model(self, layers, optimizer='adam', activation='relu', loss='mse'):
        lstm_model = Sequential()
        if(len(layers) > 2):
            lstm_model.add(LSTM(int(layers[0]*self.oldest_lag), input_shape=(
                self.oldest_lag, 1), return_sequences=True))
            for layer in layers[1:-1]:
                lstm_model.add(
                    LSTM(int(layer*self.oldest_lag), return_sequences=True))
            lstm_model.add(
                LSTM(int(layers[-1]*self.oldest_lag), activation=activation))

        elif(len(layers) == 2):
            lstm_model.add(LSTM(int(layers[0]*self.oldest_lag), input_shape=(
                self.oldest_lag, 1), return_sequences=True))
            lstm_model.add(
                LSTM(int(layers[1]*self.oldest_lag), activation=activation))

        elif(len(layers) == 1):
            lstm_model.add(LSTM(int(layers[0]*self.oldest_lag), activation=activation, input_shape=(
                self.oldest_lag, 1)))

        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer=optimizer, loss=loss)

        return lstm_model

    def train(self, model_params):
        self.model = self.create_model(**model_params)
        self.model.fit(self.training[0], self.training[1])

    def predict(self, X, future_steps):
        """
        Uses the input "X" to predict "future_steps" steps into the future for each os the instances in "X".

        :param X:
            Numpy array to make a prediction with, the shape of the input is (instances, steps).

        :param future_steps:
            Number of steps in the future to predict.

        """
        if(X.shape[1] < self.oldest_lag):
            raise Exception(
                f'''Error, to make a prediction X needs to have shape (n, {self.oldest_lag})''')

        Y_hat = np.zeros((len(X), future_steps))

        cur_X = X.copy()

        for step in range(future_steps):
            Y_hat[:, step] = np.squeeze(self.model.predict(cur_X))
            cur_X = np.roll(cur_X, -1)
            cur_X[:, -1, 0] = Y_hat[:, step]

        return Y_hat

    def auto_ml_predict(self, X, future_steps):
        X = self.automl._data_shift.transform(X)
        X = X.drop(self.index_label, axis=1)
        X = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
        y = self.predict(X, future_steps)
        return y

    def next(self, X, future_steps):
        return self.predict(self.last_x, future_steps)

    # Static Values and Methods

    # layers, optimizer='adam', activation='relu'
    # Here layers is a list of the amount of nodes in each layer. This number will be multiplied by the oldest lag being used
    params_list = [{
        "layers": [1, .7, .4],
    }, {
        "layers": [1, .5],
    }, {
        "layers": [1.2, .8, .4],
    }, {
        "layers": [1.2, 1, .7, .3],
    }, {
        "layers": [.8, .5, .3],
    }]

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):
        prefix = 'LSTM'

        print(f'Evaluating {prefix}')

        wrapper_list = []
        y_val_matrix = auto_ml._create_validation_matrix(
            cur_wrapper.validation[1].T)

        for c, params in tqdm(enumerate(LSTMWrapper.params_list)):
            auto_ml.evaluation_results[prefix+str(c)] = {}
            cur_wrapper.train(params)

            y_pred = np.array(cur_wrapper.predict(
                cur_wrapper.validation[0], max(auto_ml.important_future_timesteps)))[:, [-(n-1) for n in auto_ml.important_future_timesteps]]

            y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]
            auto_ml.evaluation_results[prefix +
                                       str(c)] = auto_ml._evaluate_model(y_val_matrix.T, y_pred)

            wrapper_list.append(copy.copy(cur_wrapper))

        return prefix, wrapper_list
