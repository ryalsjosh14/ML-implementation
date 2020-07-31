import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)
    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Read training data into pandas dataframe
    train = pd.read_csv(train_path)
    # Read both x columns
    x_train1 = np.array(train["x_1"])


    # Create x and y data for training. X data has column of ones appended for the constant
    x_train = np.column_stack((np.ones(x_train1.shape[0]), x_train1))
    y_train = np.array(train["y"])

    model = LocallyWeightedLinearRegression(.5)
    model.fit(x_train, y_train)

    eval = pd.read_csv(eval_path)
    x_eval1 = np.array(eval["x_1"])
    x_eval = np.column_stack((np.ones(x_eval1.shape[0]), x_eval1))

    y_eval = np.array(eval["y"])

    predictions = model.predict(x_eval)

    error = (y_eval - predictions).dot(y_eval - predictions) / len(y_eval)

    plt.scatter(x_train1, y_train, c= 'blue', label="Train Data")
    plt.scatter(x_eval1, predictions, c='red', label="Test Data")
    plt.legend()
    plt.show()

    # Plot validation predictions on top of training set
    # Plot data

    # plt.show()


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        #get shape of input data and initialize prediction values
        m,n = x.shape
        pred = np.zeros(m)

        #Solve for w using equation given in instructions
        #Add axes to x and self.x to make dimensions work
        w = np.exp(-np.linalg.norm(x[:,None] - self.x[None], axis=2) / (2* (self.tau)**2))

        #For each row of w, cast that row into a diagonal matrix to be multiplied
        i = 0
        for W in w:
            W = np.diag(W)

            #Calculate theta using normal equation found in part a
            den = np.dot(self.x.T,W).dot(self.x)
            num = np.dot(self.x.T, W).dot(self.y)
            theta = np.linalg.inv(den).dot(num)

            #make prediction for each of the given x values
            pred[i] = x[i].dot(theta)
            i+=1

        return pred
