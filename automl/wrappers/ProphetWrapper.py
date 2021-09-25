from .BaseWrapper import BaseWrapper
from fbprophet import Prophet
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import logging


class ProphetWrapper(BaseWrapper):
    def __init__(self, automl_instance):
        super().__init__(automl_instance)

        # passing info to warning level
        logging.getLogger('fbprophet').setLevel(logging.WARNING)

    def transform_data(self, data):
        self.data = self.automl._data_shift.transform(data)
        self.past_labels = self.automl._data_shift.past_labels
        self.future_labels = self.automl._data_shift.future_labels
        self.index_label = self.automl.index_label
        self.target_label = self.automl.target_label

        self.data.rename(columns={
            self.index_label: 'ds',
            self.target_label: 'y'
        }, inplace=True)

        # detecting the time frequency
        time_diffs = self.data['ds'][1:].values - self.data['ds'][:-1].values
        unique, counts = np.unique(time_diffs, return_counts=True)
        time_counts = dict(zip(unique, counts))

        # select the frequency with more occurences
        self.time_freq = max(time_counts, key=time_counts.get)

        train_size = int(len(self.data) * self.automl.train_val_split)

        self.training = self.data[['ds','y']].iloc[:train_size]
        self.validation = self.data.iloc[train_size:]
        self.last_x = self.validation.iloc[[-1]].copy()

    def train(self, model_params):
        # uncertainty_samples = False to speed up the prediction
        self.model = Prophet(uncertainty_samples=False, **model_params)
        self.model.fit(self.training)

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

        date_column = self.index_label if self.index_label in X.columns else 'ds'

        def step_prediction(row, future_steps):
            future_dates = pd.date_range(
                start=row[date_column],
                periods=future_steps,
                freq=pd.Timedelta(self.time_freq))

            future_df = pd.DataFrame({'ds': future_dates})

            prediction = self.model.predict(future_df).yhat

            return prediction

        Y_hat = X.apply(lambda row: step_prediction(row, future_steps), axis=1)

        Y_hat = Y_hat.to_numpy()

        return Y_hat

    def auto_ml_predict(self, X, future_steps, history):
        # date column to datetime type
        X[self.index_label] = pd.to_datetime(X[self.index_label])
        # removing timezone
        X[self.index_label] = X[self.index_label].dt.tz_localize(None)

        X.rename(columns={
            self.index_label: 'ds',
            self.target_label: 'y'
        }, inplace=True)

        y = self.predict(X, future_steps)
        return y

    def next(self, X, future_steps):
        return self.predict(self.last_x, future_steps)

    params_list = [{
        'growth': 'linear',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'seasonality_mode': 'additive',
    }, {
        'growth': 'linear',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'seasonality_mode': 'multiplicative',
    }]

    def evaluate(self):
        prefix = 'Prophet'

        print(f'Evaluating {prefix}')

        eval_list = []

        for c, params in tqdm(enumerate(ProphetWrapper.params_list)):
            self.train(params)

            y_pred = self.predict(
                self.validation[['ds','y']], max(self.automl.important_future_timesteps))

            # selecting only the important timesteps
            y_pred = y_pred[:, [-(n-1)
                                for n in self.automl.important_future_timesteps]]

            cur_eval = {
                "results": self.automl._evaluate_model(self.validation[['y']+self.future_labels].values, y_pred),
                "params": params,
                "model": copy.copy(self.model),
                "name": f'{prefix}-{c}',
                "wrapper": self
            }

            eval_list.append(cur_eval)

        return eval_list
