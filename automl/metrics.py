import numpy as np


def weighted_quantile_loss(quantile, y_true, quantile_pred):
    """
    The Weighted Quantile Loss (wQL) metric measures the accuracy of predictions
    at a specified quantile.

    :param quantile: Specific quantile to be analyzed.
    :param y_true: The observed values.
    :param quantile_pred: Quantile values that the model predicted.

    """

    # vectorize max function to apply over matrices
    max_vec = np.vectorize(max)

    first_term = quantile * max_vec(y_true - quantile_pred, 0)
    second_term = (1 - quantile) * max_vec(quantile_pred - y_true, 0)

    loss = 2 * (np.sum(first_term + second_term) / np.sum(y_true))

    return loss


def weighted_absolute_percentage_error(y_true, y_pred):
    """
    The Weighted Absolute Percentage Error (WAPE) metric measures the overall
    deviation of forecasted values from observed values.

    :param y_true: The observed values.
    :param y_pred: The predicted values.

    """

    absolute_error_sum = np.sum(np.abs(y_true - y_pred))

    loss = absolute_error_sum / np.sum(np.abs(y_true))

    return loss


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_relative_squared_error(y_true, y_pred, average=True):
    """
    Vectorized implementation of RSE
    """
    average_y_true = np.repeat(
        np.mean(y_true, axis=0, keepdims=True).T, y_true.shape[0], axis=1)
    numerator = np.sum(np.power(y_pred-y_true, 2), axis=1)
    denominator = np.sum(y_true-average_y_true, axis=1)
    if(average):
        return np.sum(np.sqrt(numerator/denominator))
    else:
        return np.sqrt(numerator/denominator)
