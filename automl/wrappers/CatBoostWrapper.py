from .BaseWrapper import BaseWrapper
from tqdm import tqdm
import copy
import numpy as np
import catboost as cat
from sklearn.model_selection import train_test_split


class CatBoostWrapper(BaseWrapper):
    def __init__(self, automl_instance):
        super().__init__(automl_instance)

    def transform_data(self, data):
        self.data = self.automl._data_shift.transform(data)
        self.past_labels = self.automl._data_shift.past_labels
        self.past_lags = self.automl._data_shift.past_lags
        self.oldest_lag = int(max(self.past_lags))
        self.index_label = self.automl.index_label
        self.target_label = self.automl.target_label
        self.last_x = data.drop(
            [self.index_label, self.target_label], axis=1).tail(1)

        X = self.data[self.past_labels]
        y = self.data[self.target_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.automl.train_val_split, shuffle=False)

        # filtering the usefull lags
        X_train = self.automl._data_shift.filter_lags(X_train)

        self.training = (X_train, y_train)
        self.validation = (X_test, y_test)

    def train(self, model_params):
        self.model = cat.CatBoostRegressor(**model_params)
        self.model.fit(self.training[0], self.training[1], verbose=False)

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

        for i, x in enumerate(X.values):
            cur_x = x.copy()
            for step in range(future_steps):
                cur_y_hat = self.model.predict(
                    self.automl._data_shift.filter_lags(cur_x).reshape(1, -1))
                Y_hat[i, step] = cur_y_hat
                cur_x = np.roll(cur_x, -1)
                cur_x[-1] = cur_y_hat

        return Y_hat

    def auto_ml_predict(self, X, future_steps, history):
        X = self.automl._data_shift.transform(X)
        X = X.drop(self.index_label, axis=1)
        y = self.predict(X, future_steps)
        return y

    def next(self, X, future_steps):
        return self.predict(self.last_x, future_steps)

    # Static Values and Methods

    params_list = [{
        'depth': 3,
        'learning_rate': 0.1,
        'l2_leaf_reg': 5,
    }, {
        'depth': 3,
        'learning_rate': 0.3,
        'l2_leaf_reg': 10,
    }, {
        'depth': 4,
        'learning_rate': 0.1,
        'l2_leaf_reg': 5,
    }, {
        'depth': 4,
        'learning_rate': 0.3,
        'l2_leaf_reg': 10,
    }, {
        'depth': 5,
        'learning_rate': 0.1,
        'l2_leaf_reg': 5,
    }, {
        'depth': 5,
        'learning_rate': 0.3,
        'l2_leaf_reg': 10,
    }, ]

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):
        prefix = 'CatBoost'

        print(f'Evaluating {prefix}')

        wrapper_list = []
        y_val_matrix = auto_ml._create_validation_matrix(
            cur_wrapper.validation[1].values.T)

        for c, params in tqdm(enumerate(CatBoostWrapper.params_list)):
            auto_ml.evaluation_results[prefix+str(c)] = {}
            cur_wrapper.train(params)

            y_pred = np.array(cur_wrapper.predict(
                cur_wrapper.validation[0], max(auto_ml.important_future_timesteps)))[:, [-(n-1) for n in auto_ml.important_future_timesteps]]

            y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]
            auto_ml.evaluation_results[prefix +
                                       str(c)] = auto_ml._evaluate_model(y_val_matrix.T, y_pred)

            wrapper_list.append(copy.copy(cur_wrapper))

        return prefix, wrapper_list
