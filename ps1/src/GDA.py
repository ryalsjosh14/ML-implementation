import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    # *** START CODE HERE ***
    train = pd.read_csv(train_path)

    # Read both x columns
    x_train1 = np.array(train["x_1"])
    x_train2 = np.array(train["x_2"])

    # Create x and y data for training. X data does NOT have column of ones for GDA
    x_train = np.column_stack((x_train1, x_train2))
    y_train = np.array(train["y"])

    # create instance of GDA model
    model = GDA();

    # Fit GDA parameters to the training data
    model.fit(x_train, y_train)

    # Read evaluation data and seperate based on category (y=0, y=1)
    eval = pd.read_csv(eval_path)
    true_vals = eval.loc[eval['y'] == 1.0]
    false_vals = eval.loc[eval['y'] == 0.0]

    # Seperate the x1 and x2 values for both categories
    x_test1_true = np.array(true_vals["x_1"])
    x_test2_true = np.array(true_vals["x_2"])

    x_test1_false = np.array(false_vals["x_1"])
    x_test2_false = np.array(false_vals["x_2"])

    # Get all test values, not seperated by category
    x_test1 = np.array(eval["x_1"])
    x_test2 = np.array(eval["x_2"])
    x_test = np.column_stack((np.ones(x_test1.shape[0]), x_test1, x_test2))
    y_test = np.array(eval["y"])

    # Predict categories of test data
    predictions = model.predict(x_test)

    # Produce a scatter plot of the test data, labeled by category
    plt.scatter(x_test1_true, x_test2_true, label="y=1")
    plt.scatter(x_test1_false, x_test2_false, label="y=0")

    # Create domain for decision boundary line
    domain = np.linspace(0, 7, 100)

    # Solve for x2 by setting the weighted sum of input equal to 0,
    # and using the domain as the x1
    num = - (model.theta[0] + np.dot(model.theta[1], domain))
    den = model.theta[2];
    x2 = num / den

    # Plot the decision boundary
    plt.plot(domain, x2, label="Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.savefig("GDA_dataset1.png", bbox_inches="tight")

    plt.show()

    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        #Get shape of input x (without column of 1's
        m,n = x.shape

        #Solve for parameters mu0, mu1 and phi
        phi = sum(y==1)/m
        mu_zero = sum(x[y==0]) / sum(y==0)
        mu_one = sum(x[y==1]) / sum(y==1)

        #Solve for parameter sigma
        difference = x.copy()
        difference[y==0] -= mu_zero
        difference[y==1] -= mu_one
        sigma = (1/m) * difference.T.dot(difference)

        #Solve for theta and theta0 using equation from 1c
        sigma_inverse = np.linalg.inv(sigma)
        theta = - sigma_inverse.dot((mu_zero - mu_one))
        theta_zero = -(1/2) * (mu_one.T.dot(sigma_inverse).dot(mu_one) - mu_zero.T.dot(sigma_inverse).dot(mu_zero)) - math.log( (1-phi)/phi)

        #Turn theta_zero from a number to a numpy array
        theta_zero = np.array(theta_zero)

        #Combine theta and theta zero
        self.theta = np.hstack((theta_zero, theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        # Calculate predicted values using sigmoid equation
        pred = 1 / (1 + np.exp(-1 * np.dot(x, self.theta)) )
        # Return nparray of predicted probabilities
        return pred
        # *** END CODE HERE