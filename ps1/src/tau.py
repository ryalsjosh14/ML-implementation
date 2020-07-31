import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from WeightedLinear import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.
    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Read training data
    train = pd.read_csv(train_path)
    x_train1 = np.array(train["x_1"])
    x_train = np.column_stack((np.ones(x_train1.shape[0]), x_train1))
    y_train = np.array(train["y"])

    #Read evaluate data
    eval = pd.read_csv(valid_path)
    x_eval1 = np.array(eval["x_1"])
    x_eval = np.column_stack((np.ones(x_eval1.shape[0]), x_eval1))
    y_eval = np.array(eval["y"])

    #initialize min tau and min error values
    min_tau = tau_values[0]
    min_error = float("inf")

    #For each tau value, calculate the MSE and find the tau with minimum MSE
    for tau in tau_values:
        #Create model with specified tau value
        model = LocallyWeightedLinearRegression(tau)
        #Fit model with train data
        model.fit(x_train, y_train)
        #Predict values of evaluation data
        predictions = model.predict(x_eval)
        #Calculate error for given tau value
        error = (y_eval - predictions).dot(y_eval - predictions) / len(y_eval)
        print("tau: {}, error: {}".format(tau,error))
        #IF error is minimum, place it as min error
        if error < min_error:
            min_error = error
            min_tau = tau

        plt.scatter(x_train1, y_train, c='blue', label="Train Data")
        plt.scatter(x_eval1, predictions, c='red', label="Test Data")
        plt.legend()
        plt.title("tau: {}".format(tau))
        plt.show()

    print("Minimum Tau: {}".format(min_tau))
    #Calculate MSE of minimum tau value on test data
    test = pd.read_csv(test_path)
    x_test1 = np.array(test["x_1"])
    # Create x and y data for training. X data has column of ones appended for the constant
    x_test = np.column_stack((np.ones(x_test1.shape[0]), x_test1))
    y_test = np.array(test["y"])

    model = LocallyWeightedLinearRegression(min_tau)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    error = (y_test - predictions).dot(y_test - predictions) / len(y_test)
    print("final error: {}".format(error))

    # *** END CODE HERE ***