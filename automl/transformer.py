from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import pacf
import numpy as np
import pandas as pd

class DataShift(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.05, nlags=24, past_lags=None):
        self.past_labels = []
        self.threshold = threshold
        self.nlags = nlags
        self.past_lags = past_lags
        self.past_labels = []
        self._target_label = ''

    def _calculate_past_lags(self, data):
        """
        Find the best number of past lags using the Partial Autocorrelation Function.

        :param threshold:
            Confidence interval threshold.

        :param nlags:
            Number of possible lags considered.

        """

        # apply the pacf over the data
        # exclude the first value that is the observed value
        corr = pacf(data, nlags=self.nlags)[0:]

        # filter by the threshold
        corr_filtered = np.argwhere((corr > self.threshold) | (corr < -self.threshold))
        
        past_lags = np.hstack(corr_filtered)

        return past_lags

    def _data_shift(self, data, past_lags):
        """
        Shift the data by past lags inserting the past time steps as new columns.

        :param data:
            Data that will be shift by past lags.

        :param past_lags: 
            List with the past lags indices that will be inserted.
            Example: past_lags = [0, 2, 4] will insert the columns of time steps
            t-1, t-3 and t-5

        """
        X = []

        # max number of lags possible
        max_lags = max(past_lags) + 1

        idx = max_lags
        while True:
            new_line = data.iloc[idx - max_lags : idx, 1].values
            
            X.append(new_line)
            
            idx += 1

            if idx >= len(data): break

        # label of each past lag going from -max_lags to -min_lags
        past_labels = ['target_' + str(-(i))
                       for i in range(max_lags, 0, -1)]

        data = data.iloc[max_lags:, :]
        data.loc[:, past_labels] = X

        return data, past_labels

    def filter_lags(self, data):

        # if it is a dataframe filter the selected lags by columns
        if isinstance(data, pd.DataFrame):
            # label of each past lag going from -max_lags to -min_lags
            useful_past_labels = ['target_' + str(-(i + 1))
                        for i in reversed(self.past_lags)]
            return data[useful_past_labels]

        # if it is an array filter by position
        elif isinstance(data, np.array):
            return data[self.past_lags]

    def fit(self, data):
        _, self._target_label = tuple(data.columns)

        # find the best past lags value
        if self.past_lags is None:
            self.past_lags = self._calculate_past_lags(data[self._target_label])

        return self

    def transform(self, X):
        
        # add to the data the past lags
        X_, self.past_labels = self._data_shift(X, self.past_lags)

        return X_