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
        self.future_labels = self.automl._data_shift.future_labels
        self.past_lags = self.automl._data_shift.past_lags
        self.oldest_lag = int(max(self.past_lags)) + 1
        self.index_label = self.automl.index_label
        self.target_label = self.automl.target_label
        self.last_x = data.drop(
            [self.index_label, self.target_label], axis=1).tail(1).copy()

        X = self.data[self.past_labels]
        y = self.data[[self.target_label] + self.future_labels]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.automl.train_val_split, shuffle=False)

        # filtering the usefull lags
        X_train = self.automl._data_shift.filter_lags(X_train)

        X_train = np.reshape(
            X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(
            X_test.values, (X_test.shape[0], X_test.shape[1], 1))

        self.training = (X_train, y_train[self.target_label])
        self.validation = (X_test, y_test)

    def create_model(self, layers, optimizer='adam', activation='relu', loss='mse'):
        lstm_model = Sequential()
        if(len(layers) > 2):
            lstm_model.add(LSTM(int(layers[0]*len(self.past_lags)), input_shape=(
                len(self.past_lags), 1), return_sequences=True))
            for layer in layers[1:-1]:
                lstm_model.add(
                    LSTM(int(layer*len(self.past_lags)), return_sequences=True))
            lstm_model.add(
                LSTM(int(layers[-1]*len(self.past_lags)), activation=activation))

        elif(len(layers) == 2):
            lstm_model.add(LSTM(int(layers[0]*len(self.past_lags)), input_shape=(
                len(self.past_lags), 1), return_sequences=True))
            lstm_model.add(
                LSTM(int(layers[1]*len(self.past_lags)), activation=activation))

        elif(len(layers) == 1):
            lstm_model.add(LSTM(int(layers[0]*len(self.past_lags)), activation=activation, input_shape=(
                len(self.past_lags), 1)))

        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer=optimizer, loss=loss)

        return lstm_model

    def train(self, model_params):
        self.model = self.create_model(**model_params)
        self.model.fit(self.training[0], self.training[1])

    def clear_excess_data(self):
        del self.data
        del self.training
        del self.validation

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
            to_predict = cur_X[:, self.past_lags, :]
            Y_hat[:, step] = np.squeeze(self.model.predict(to_predict))
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

    def evaluate(self):
        prefix = 'LSTM'

        print(f'Evaluating {prefix}')

        eval_list = []

        for c, params in tqdm(enumerate(LSTMWrapper.params_list)):
            self.train(params)

            y_pred = np.array(self.predict(
                self.validation[0], max(self.automl.important_future_timesteps)))[:, [-(n-1) for n in self.automl.important_future_timesteps]]

            cur_eval = {
                "results": self.automl._evaluate_model(self.validation[1].values, y_pred),
                "params": params,
                "model": copy.copy(self.model),
                "name": f'{prefix}-{c}',
                "wrapper": self
            }

            eval_list.append(cur_eval)

        return eval_list
