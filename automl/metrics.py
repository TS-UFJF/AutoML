import numpy as np

# When inplementing new metrics remember that AutoML passes a 2D array
# with shape [instances, timesteps]


def weighted_absolute_percentage_error(y_true, y_pred, average=True):
    """
    The Weighted Absolute Percentage Error (WAPE) metric measures the overall
    deviation of forecasted values from observed values.

    :param y_true: The observed values.
    :param y_pred: The predicted values.
    :param average: If average is set to True, returns the average of all instances,
        otherwise, returns an array of the WAPEs of all instances.

    """

    absolute_error_sum = np.sum(np.abs(y_true - y_pred), axis=1)

    loss = absolute_error_sum / np.sum(np.abs(y_true), axis=1)

    if(average):
        return np.mean(loss)
    else:
        return loss


def mean_absolute_percentage_error(y_true, y_pred, average=True):
    mape = np.mean(np.abs((y_true - y_pred) / y_true), axis=1) * 100
    if(average):
        return np.mean(mape)
    else:
        return mape


def root_relative_squared_error(y_true, y_pred, average=True):
    """
    Vectorized implementation of RSE
    """
    average_y_true = np.repeat(
        np.mean(y_true, axis=0, keepdims=True).T, y_true.shape[0], axis=1)
    numerator = np.sum(np.power(y_pred-y_true, 2), axis=1)
    denominator = np.sum(y_true-average_y_true, axis=1)
    if(average):
        return np.mean(np.sqrt(numerator/denominator))
    else:
        return np.sqrt(numerator/denominator)
