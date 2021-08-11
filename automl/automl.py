import pandas as pd
import numpy as np
import warnings
import gc
import pprint
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from .metrics import *
from .transformer import DataShift
from .wrappers.LightGBMWrapper import LightGBMWrapper


class AutoML:
    def __init__(self, path, jobs=0, fillnan='ffill', nlags=24, wrapper_constructors=[LightGBMWrapper], important_future_timesteps=[1], train_val_split=0.8):
        """
        AutoML is an auto machine learning project with focus on predict
        time series using simple usage and high-level understanding over
        time series prediction methods.

        :param path:
            Path to the input csv. The csv must be in the format (date, value).

        :param jobs:
            Number of jobs used during the training and evaluation.

        :param wrapper_constructors:
            List with the constructors of all the wrappers that will be used in this instance of AutoML.

        :param fillnan: {'bfill', 'ffill'}, default ffill
            Method to use for filling holes.
            ffill: propagate last valid observation forward to next valid.
            bfill: use next valid observation to fill gap.

        :param nlags: default 24
            Maximum number of hours in the past used for each prediction

        :param important_future_timesteps: default [1]
            Timesteps in the future that will be used in model selection
        """

        warnings.filterwarnings('ignore')

        self.path = path
        self.jobs = jobs
        self.fillnan = fillnan
        self.nlags = nlags

        self.data = pd.read_csv(self.path)
        self.target_label = None
        self.index_label = None
        self.oldest_lag = 1
        self.train_val_split = train_val_split
        self.wrapper_constructors = wrapper_constructors
        self.wrappers = {wr.__name__: wr(self) for wr in wrapper_constructors}
        self.important_future_timesteps = important_future_timesteps

        if len(self.data.columns) > 2:
            raise Exception('Data has more than 2 columns.')

        self._data_shift = DataShift(nlags=self.nlags)

        # results obtained during evaluation
        self.evaluation_results = []

        # the chosen model
        self.model = None
        self.model_name = ''
        self.quantile_models = []

        self._transform_data()
        self._evaluate()

    def _transform_data(self):
        """
        Clean and prepare data to use on each wrapper.

        """

        self.treated_data = self.data.copy()

        self.index_label, self.target_label = tuple(self.treated_data.columns)

        # date column to datetime type
        self.treated_data[self.index_label] = pd.to_datetime(
            self.treated_data[self.index_label])
        # removing timezone
        self.treated_data[self.index_label] = self.treated_data[self.index_label].dt.tz_localize(
            None)

        # find the best past lags value
        self._data_shift.fit(self.treated_data)
        self.oldest_lag = int(max(self._data_shift.past_lags)) + 1

    def _evaluate_model(self, y_true, y_pred):
        """
        Evaluate a specific model given the data to be tested.

        :param model: Model to be evaluated.
        :param y_pred: predicted values, with shape [instances, time_steps].
        :param y_true: y values that represents the real values, with shape [instances, time_steps].

        """

        results = {}

        results['Mean WAPE'] = weighted_absolute_percentage_error(
            y_true, y_pred)
        results['Mean RMSE'] = mean_squared_error(
            y_true, y_pred, squared=False)
        results['Mean MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
        results['Mean RSE'] = root_relative_squared_error(y_true, y_pred)
        results['Mean MASE'] = mean_absolute_scaled_error(y_true, y_pred)
        results['Mean MAE'] = mean_absolute_error(y_true, y_pred)
        results['Mean sMAPE'] = symmetric_mean_absolute_percentage_error(
            y_true, y_pred)
        results['Mean msMAPE'] = msMAPE(y_true, y_pred)
        results['Mean WAPE'] = weighted_absolute_percentage_error(
            y_true, y_pred)
        results['Median RMSE'] = np.median(mean_squared_error(
            y_true, y_pred, squared=False, multioutput='raw_values'))
        results['Median MAPE'] = np.median(mean_absolute_percentage_error(
            y_true, y_pred, multioutput='raw_values'))
        results['Median RSE'] = np.median(root_relative_squared_error(
            y_true, y_pred, multioutput='raw_values'))
        results['Median MASE'] = np.median(mean_absolute_scaled_error(
            y_true, y_pred, multioutput='raw_values'))
        results['Median MAE'] = np.median(mean_absolute_error(
            y_true, y_pred, multioutput='raw_values'))
        results['Median sMAPE'] = np.median(symmetric_mean_absolute_percentage_error(
            y_true, y_pred, multioutput='raw_values'))
        results['Median msMAPE'] = np.median(
            msMAPE(y_true, y_pred, multioutput='raw_values'))

        return results

    def _create_validation_matrix(self, val_y):
        """
        Function Creating the validation matrix.
        """
        y_val = []

        for n in self.important_future_timesteps:
            y_val.append(np.roll(val_y, -(n - 1)).T)

        return np.array(y_val)[:,
                               :-(max(self.important_future_timesteps))]

    def _evaluate(self):
        """
        Compare baseline models

        """
        # Vamos fazer com os modelos sempre usando a api do Scikit Learn pq a gnt vai usar ele para o RandomSearch

        for cur_wrapper in self.wrappers.values():
            cur_wrapper.transform_data(self.treated_data.copy())
            eval_list = cur_wrapper.evaluate()
            self.evaluation_results += eval_list
            cur_wrapper.clear_excess_data()
            gc.collect()

        # Ordering the models acording to the results and setting the best model as the current model
        self.evaluation_results.sort(key=lambda item: item["results"]["Mean WAPE"])
        self.set_evaluated_model(0)

    def print_evaluation_results(self):
        """
        Prints the names, parameters and results of evaluation_results.
        """
        for evaluation in self.evaluation_results:
            print("Model: ", evaluation['name'])
            print("Results: ")
            pprint.pprint(evaluation['results'])
            print("Parameters: ")
            pprint.pprint(evaluation['params'])
            print("-----------------------------")

    def predict(self, X, future_steps, history=[]):
        """
        Uses the input "X" to predict "future_steps" steps into the future.

        :param X:
            Values to make a prediction with.

        :param future_steps:
            Number of steps in the future to predict.

        :param history:
            History buffer that will be used to as base to new predictions.
            The history based models demands at least 2 * oldest_lag to make the predictions,
            so if the choosen model is one of them it demands this parameter.
            History based models: [TFT].

        """
        if(len(X) < self.oldest_lag):
            raise Exception(f'''Error, to make a prediction X needs to be at
                                least {self.oldest_lag} items long''')
        y = self.model.auto_ml_predict(X, future_steps, history)

        return y

    def set_evaluated_model(self, evaluation_results_index: int):
        """
        Sets the "evaluation_results_intex"-th item of self.evaluation_results as the model currently
        being used by AutoML

        :param evaluation_results_index:
            The index in self.evaluation_results of the model being set.
        """
        self.evaluation_results[evaluation_results_index]["wrapper"].model = self.evaluation_results[evaluation_results_index]["model"]
        self.model = self.evaluation_results[evaluation_results_index]["wrapper"]
        self.model_name = self.evaluation_results[evaluation_results_index]["name"]

    def next(self, future_steps):
        """
        Predicts the next "future_steps" steps into the future using the data inserted for training.

        :param future_steps:
            Number of steps in the future to predict.

        """
        return self.model.next(self.data, future_steps)
