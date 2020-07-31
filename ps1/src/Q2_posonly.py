import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LogisticRegression import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'

"""TODO: Figure out how to appy alpha properly"""


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.
    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.
    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    #Part c, logistic regression on true (t) values
    model = LogisticRegression()

    # Read training data into pandas dataframe
    train = pd.read_csv(train_path)

    # Read both x columns
    x_train1 = np.array(train["x_1"])
    x_train2 = np.array(train["x_2"])

    # Create x and y data for training. X data has column of ones appended for the constant
    x_train = np.column_stack((np.ones(x_train1.shape[0]), x_train1, x_train2))
    t_train = np.array(train["t"])

    #Fit the logistic regression model with the x and t data
    model.fit(x_train, t_train)

    #Get test data
    test = pd.read_csv(test_path)
    x_test1 = np.array(test["x_1"])
    x_test2 = np.array(test["x_2"])

    # Create domain for decision boundary line
    domain = np.linspace(-6, 3, 124)

    # Solve for x2 by setting the weighted sum of input equal to 0,
    # and using the domain as the x1
    num = - (model.theta[0] + np.dot(model.theta[1], domain))
    den = model.theta[2];
    x2_c = num / den

    #Get test x values
    x_test = np.column_stack((np.ones(x_test1.shape[0]), x_test1, x_test2))

    #Make predidctions based on the model parameters
    predictions = model.predict(x_test)

    np.savetxt(pred_path_c, predictions)


    #Part d, use y-labels only
    y_train = np.array(train["y"])

    #Fit the model with just the y values
    model.fit(x_train, y_train)

    #Make new predictions based y data
    predictions = model.predict(x_test)

    np.savetxt(pred_path_d, predictions)

    # Solve for x2 by setting the weighted sum of input equal to 0,
    # and using the domain as the x1
    num = - (model.theta[0] + np.dot(model.theta[1], domain))
    den = model.theta[2];
    x2_d = num / den

    #Part e,estimate alpha and re-do predictions
    validate = pd.read_csv(valid_path)

    x1_true = x_test1[test["y"] == 1]
    x1_false = x_test1[test["y"] == 0]
    x2_true = x_test2[test["y"] == 1]
    x2_false = x_test2[test["y"] == 0]

    x_valid1 = np.array(validate["x_1"])
    x_valid2 = np.array(validate["x_2"])
    y_valid = np.array(validate["y"])

    x2_valid_true = x_valid2[validate["y"] == 1]
    x1_valid_true = x_valid1[validate["y"] == 1]


    x_valid = np.column_stack((np.ones(x_valid1.shape[0]), x_valid1, x_valid2))

    predictions = model.predict(x_valid)

    alpha = sum(predictions[y_valid==1]) / len(validate.loc[validate["y"]==1])

    new_predictions = predictions / alpha



    #Adjust decision boundary line from part d by including alpha in calcualation
    num = - (model.theta[0] + np.dot(model.theta[1], domain))
    den = model.theta[2];
    x2_e = (num) / den

    #Plots

    #Scatter test data, labeled as either t=1 or t=0
    plt.scatter(x1_true, x2_true, label="t=1")
    plt.scatter(x1_false, x2_false, label="t=0")

    # Plot the decision boundaries
    plt.plot(domain, x2_d, label="Decision Boundary pt. d")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    #Plot 2 (part d)
    plt.scatter(x1_true, x2_true, label="t=1")
    plt.scatter(x1_false, x2_false, label="t=0")
    plt.plot(domain, x2_c, label="Decision Boundary pt. c")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    # Plot 3 (part e)
    plt.scatter(x1_true, x2_true, label="t=1")
    plt.scatter(x1_false, x2_false, label="t=0")
    plt.plot(domain, x2_e, label="Decision Boundary pt. e")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

