import matplotlib.pyplot as plt
import numpy as np
import os
import math


PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # x (m, K)

    m = x.shape[0]

    indices = np.arange(m)
    np.random.shuffle(indices)
    groups = np.array_split(x[indices], K, axis = 0)

    mu = []
    sigma = []
    for group in groups:
        mu_temp = np.mean(group, axis = 0)
        mu.append(mu_temp)
        sigma.append((x-mu_temp).T.dot(x-mu_temp) / group.shape[0])

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = (1/K) * np.ones(K)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = (1/K) * np.ones((m, K))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """

    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll-prev_ll) >= eps):

         # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        #Get shape values
        m, n = x.shape
        k = phi.shape[0]
        #For each Gaussian, adjust w by multiplying the Gaussian distribution by the probability of z
        for j in range(k):
            w[:, j] = Gaussian(x, sigma[j], mu[j]) * phi[j]

        #Normalize w by dividing by the sum of each row (so each row sums to 1)
        w = w / w.sum(axis=1, keepdims=True)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        #phi[j] is the average probability of a random point being in Gaussian j
        phi = w.sum(axis=0) / m

        #update mu and j
        for j in range(k):
            mu[j] = w[:, j].dot(x) / sum(w[:,j])
            sigma[j] = (x-mu[j]).T.dot(np.diag(w[:,j])).dot(x-mu[j]) / sum(w[:, j])


        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.


        p_z = np.zeros(m)
        for j in range(k):

            p_z += Gaussian(x, sigma[j], mu[j]) * phi[j]

        prev_ll = ll
        ll = np.sum(np.log(p_z))
        print('Iteration #: {}, Log Likelihood: {}'.format(it, ll))
        it += 1

        # *** END CODE HERE ***
    print(w)
    return w


def Gaussian(x, sigma, mu):
    """

    :param x: (m,n)
    :param sigma: (n,n)
    :param mu: (n)
    :return: Gaussian distribution
    """
    m, n = x.shape

    x_T = x[:, None, :]
    mu_T = mu[None, :]
    x = x[:,:, None]
    mu = mu[:,None]

    exp_term = -(1/2) * np.matmul( (x_T-mu_T), np.matmul(np.linalg.pinv(sigma), (x-mu)) )[:,0,0]
    coefficient_term = 1 / ( np.power((2 * math.pi),(n/2)) * math.sqrt(np.linalg.det(sigma)) )
    return coefficient_term * np.exp(exp_term)


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        # Get Gaussian
        m, n = x.shape
        m_tilde = x_tilde.shape[0]
        k = phi.shape[0]


        w_tilde = np.zeros((m_tilde, k))

        # print(phi.shape)
        for j in range(k):
            w[:, j] = Gaussian(x, sigma[j], mu[j]) * phi[j]
            w_tilde[:, j] = (z==j).squeeze()

        w = w / w.sum(axis=1, keepdims=True)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = (w.sum(axis=0) + alpha * w_tilde.sum(axis=0))
        phi = phi / phi.sum()

        for j in range(k):
            mu[j] = ((w[:, j].dot(x) + (alpha*w_tilde[:,j]).dot(x_tilde) ) / (sum(w[:,j]) + alpha*w_tilde[:,j].sum(axis=0)))
            sigma[j] = ( (x-mu[j]).T.dot(np.diag(w[:,j])).dot(x-mu[j]) + (alpha * (x_tilde-mu[j]).T.dot(np.diag(w_tilde[:,j])).dot((x_tilde - mu[j]))) ) / (sum(w[:, j]) +(alpha*w_tilde[:,j].sum()))


        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        p_z = np.zeros(m)
        for j in range(k):
            p_z += Gaussian(x, sigma[j], mu[j]) * phi[j]

        p_z_tilde = np.zeros(m_tilde)
        for j in range(k):
            p_z_tilde += Gaussian(x_tilde, sigma[j], mu[j]) * phi[j]

        prev_ll = ll
        ll = np.sum(np.log(p_z)) + alpha * np.sum(np.log(p_z_tilde))
        print('Iteration #: {}, Log Likelihood: {}'.format(it, ll))
        it += 1
        # *** END CODE HERE ***

    print(w)
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)
    plt.show()


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
