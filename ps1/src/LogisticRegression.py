import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from linear_model import LinearModel
import pandas as pd

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    #Read training data into pandas dataframe
    train = pd.read_csv(train_path)
    #Read both x columns
    x_train1 = np.array(train["x_1"])
    x_train2 = np.array(train["x_2"])

    #Create x and y data for training. X data has column of ones appended for the constant
    x_train = np.column_stack((np.ones(x_train1.shape[0]),x_train1, x_train2))
    y_train = np.array(train["y"])


    # *** START CODE HERE ***
    #create instance of linear regression model
    model = LogisticRegression();

    #Fit logistic curve to the training data
    model.fit(x_train, y_train)

    #Read evaluation data and seperate based on category (y=0, y=1)
    eval = pd.read_csv(eval_path)
    true_vals = eval.loc[eval['y'] == 1.0]
    false_vals = eval.loc[eval['y'] == 0.0]

    # Seperate the x1 and x2 values for both categories
    x_test1_true = np.array(true_vals["x_1"])
    x_test2_true = np.array(true_vals["x_2"])

    x_test1_false = np.array(false_vals["x_1"])
    x_test2_false = np.array(false_vals["x_2"])

    #Get all test values, not seperated by category
    x_test1 = np.array(eval["x_1"])
    x_test2 = np.array(eval["x_2"])
    x_test = np.column_stack((np.ones(x_test1.shape[0]), x_test1, x_test2))
    y_test = np.array(eval["y"])

    #Predict categories of test data
    predictions = model.predict(x_test)

    #Produce a scatter plot of the test data, labeled by category
    plt.scatter(x_test1_true, x_test2_true, label="y=1")
    plt.scatter(x_test1_false, x_test2_false, label="y=0")

    #Create domain for decision boundary line
    domain = np.linspace(0,7,100)

    #Solve for x2 by setting the weighted sum of input equal to 0,
    #and using the domain as the x1
    num = - (model.theta[0] + np.dot(model.theta[1],domain))
    den = model.theta[2];
    x2 = num / den

    #Plot the decision boundary
    plt.plot(domain, x2, label="Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    plt.savefig("Logistic_dataset1.png")

    #Show the plot
    plt.show()

    np.savetxt(pred_path, predictions)

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """

        m,n = x.shape

        #Initialize theta
        self.theta = np.zeros(n)

        while True:
            theta = self.theta

            #Calculate the prediction using the sigmoid function
            h = 1 / (1+ np.exp(-1 * np.dot(x,theta)))

            #Calculate the gradient
            grad = (1/m) * ( np.dot( (h-y), x) )
            #Calculate the Hessian
            H = (1/m) * (h.dot(1-h) * (x.T.dot(x)) )

            #Calculate inverse of the hessian
            inverse_H = np.linalg.inv(H)

            #Update theta using Newton's method equation
            theta_new = theta - ( inverse_H.dot(grad) )

            #If changes have become less than the eps, break and use theta as model
            if np.linalg.norm(theta_new - theta, ord=1) < self.eps:
                self.theta = theta_new
                break

            self.theta = theta_new


    def predict(self, x):
        """Make a prediction given new inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        #Calculate predicted values using sigmoid equation
        pred = 1 / (1+np.exp(-1 * np.dot(x, self.theta)))
        #Return nparray of predicted probabilities
        return pred

