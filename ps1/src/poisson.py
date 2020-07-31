import numpy as np
import pandas as pd


from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.
    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Read training data into pandas dataframe
    train = pd.read_csv(train_path)
    # Read both x columns
    x_train1 = np.array(train["x_1"])
    x_train2 = np.array(train["x_2"])
    x_train3 = np.array(train["x_3"])
    x_train4 = np.array(train["x_4"])



    # Create x and y data for training. X data has column of ones appended for the constant
    x_train = np.column_stack((np.ones(x_train1.shape[0]), x_train1, x_train2, x_train3, x_train4))
    y_train = np.array(train["y"])


    # *** START CODE HERE ***
    #Create poisson model
    model = PoissonRegression()
    model.fit(x_train, y_train, lr)
    #print(model.theta)

    #Get eval data
    eval = pd.read_csv(eval_path)
    x_eval1 = np.array(eval["x_1"])
    x_eval2 = np.array(eval["x_2"])
    x_eval3 = np.array(eval["x_3"])
    x_eval4 = np.array(eval["x_4"])

    x_eval = np.column_stack((np.ones(x_eval1.shape[0]), x_eval1, x_eval2, x_eval3, x_eval4))
    y_eval = np.array(eval["y"])

    #Predict values and save
    predictions = model.predict(x_eval)
    print(predictions)
    np.savetxt(pred_path, predictions)

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y, lr):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m,n = x.shape

        self.theta = np.zeros(n)

        while True:

            theta = self.theta

            #Solve for gradient of poisson model
            grad = (1/m) * (y - np.exp(x.dot(theta))).dot(x)
            #Update theta using stochastic update rule
            theta_new = theta + (lr * grad)

            # If changes have become less than the eps, break and use theta as model
            if np.linalg.norm(theta_new - theta, ord=1) < self.eps:
                self.theta = theta_new
                break

            self.theta = theta_new


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        #Predict values based on poisson natrual poisson prediction (natural paramter = exp(lambda))
        return np.exp(x.dot(self.theta))

        # *** END CODE HERE ***