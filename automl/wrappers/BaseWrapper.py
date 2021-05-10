class BaseWrapper:
    """
    This class exists to give an exemple of what is essential for a 
    wrapper for this package.
    """

    def __init__(self, automl_instance):
        self.automl = automl_instance
        self.quantiles = self.automl.quantiles

    def transform_data(self, data):
        """
        Recieves all of the data in a Dataframe
        - The name of the time column is in 'self.automl.index_label'
        - The name of the column with the value we want to predict is in 'self.automl.target_label'
        - All the other columns with the numerical lags values is in 'self.automl._data_shift.past_labels'

        The purpose of this method is to make the necessary preparations in this data, and,
        if needed, split in training and validation (the chosen split is in 'self.automl.train_val_split'
        and is a number between 0 and 1) and X and Y.

        This method doesn't return anything and all modifications should happen to this classe's atributes.

        """
        pass

    def train(self, model_params, quantile_params):
        """
        Recieves the parameters necessary to train a model, these parameters should be enough to train a model
        able to predict a value, or the quantiles in 'self.quantiles'.

        This method will be ran many times when using the _evaluate method, it should train the models and save 
        them as attributes of the object, in order for them to be evaluated.

        This method doesn't return anything and all modifications should happen to this classe's atributes.
        """
        pass

    def predict(self, X, future_steps, quantile=False):
        """
        Recieves the matrix 'X' with shape (instances, steps), and for each instance, it should make the prediction
        of the next 'future_steps' of the series, this predicton can be for the values or the quantiles, depending on
        the value of 'quantile'.

        This method will make the predictions for each of the instances, then, create and return 'Y_hat' with the values 
        predicted, 'Y_hat' should have the following shape: (instances, future_steps) or (instances, future_steps, quantile),
        depending on the value of 'quantile'

        This method returns 'Y_hat'.
        """
        Y_hat = np.zeros((len(X), future_steps, len(self.quantiles))
                         ) if quantile else np.zeros((len(X), future_steps))

        return Y_hat

    def auto_ml_predict(self, X, future_steps, quantile, history):
        """
        This will be the predict used by the end user when using auto_ml. 'X' will be a series and this function will return the next
        'future_steps' values of this series, this prediction will return values or quantiles depending on 'quantile', and if needed
        'history' contains 2 * oldest_lag previous values of the series

        This method return the the prediction 'y' as a np.array of length 'future_steps'.
        """
        pass

    def next(self, future_steps, quantile=False):
        """
        This method uses the last values of the data used for training, and predicts the next 'future_steps' values, 
        this prediction will return values or quantiles depending on 'quantile'.

        This method return the the prediction 'y' as a np.array of length 'future_steps'.
        """
        pass

    # Static Values and Methods

    params_list = []
    """
    This static attribute contains a list of testing values for _evaluate. Each item of this list should be a dictionary with 
    the arguments that will be evaluated by the method.
    """

    quantile_params = {}
    """
    If needed, this static attribute may be used to set the parameters needed to turn the regular model, in a model to predict the quantile values.
    """

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):
        """
        This static method recieves the instance of 'auto_ml' that is being executed and 'cur_wrapper' that is the instance of this wrapper being evaluated by auto_ml.

        This method should train 'cur_wrapper' throung all the parameters in 'params_list' and evaluate-it along the way.
        It should add the results of each train as
        'evaluation_results[prefix + str(index of the params in 'params_list')][str(quantile) for the quantile evaluations and 'default' for the regular evaluation]

        This method should return (prefix, wrapper_list), where prefix is the name of the model, and wrapper_list is a list of copies of 'cur_wrapper'
        after being trained for each of the parameter dictionaries in 'params_list'.
        """
        prefix = 'BaseWrapper'  # Prefix should be the name of the model being evaluated

        print(f'Evaluating {prefix}')

        wrapper_list = []

        return prefix, wrapper_list
