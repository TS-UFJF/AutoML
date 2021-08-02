import numpy as np
import warnings


def _process_multioutput(arr, multioutput):
    if(multioutput == 'uniform_average'):
        return np.mean(arr)
    if(multioutput == 'raw_values'):
        return arr
    if(isinstance(multioutput, list) and not isinstance(multioutput, str)):
        if((np.squeeze(np.array(multioutput)).shape[0] == len(arr)) and np.squeeze(np.array(multioutput)).ndim == 1):
            np.average(arr, weights=np.squeeze(np.array(multioutput)))
        else:
            warnings.warn(f"Expected shape {arr.shape} from 'multioutput' and received shape {np.array(multioutput).shape}" +
                          ", returning an 'uniform_average'")
            return np.mean(arr)
    else:
        print("The value in multioutput is invalid, returning an 'uniform_average'")
        return np.mean(arr)


# When inplementing new metrics remember that AutoML passes a 2D array
# with shape [instances, timesteps]


def weighted_absolute_percentage_error(y_true, y_pred, multioutput='uniform_average'):
    """
    The Weighted Absolute Percentage Error (WAPE) metric measures the overall
    deviation of forecasted values from observed values.

    :param y_true: The observed values.
    :param y_pred: The predicted values.
    :param multioutput: can be {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    """

    absolute_error_sum = np.sum(np.abs(y_true - y_pred), axis=1)

    loss = absolute_error_sum / np.sum(np.abs(y_true), axis=1)
    return _process_multioutput(loss, multioutput)


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
