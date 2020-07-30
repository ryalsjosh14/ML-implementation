from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

def k_means(points, K):
    """
    Arguments:
        - points: (M, 3)
    """
    m, n = points.shape

    #Initialize mu randomly
    #mu (m,3)
    mu = np.zeros((K,n))
    for i in range(K):
        random_row = np.random.randint(m)
        mu[i] = points[random_row][:]

    eps = 1e-3  # Convergence threshold
    max_iter = 300
    it = 0
    prev_c, c = None, None

    while it < max_iter and (prev_c is None or np.abs(c - prev_c).mean() > eps):

        prev_c = c
        #w (m, K)
        w = np.linalg.norm(points[:, None, :] - mu[None, :, :], axis=2)
        #c (m,)
        c = np.argmin(w, axis=1)

        #reset mu
        for j in range(K):
            mu[j] = points[c == j].mean(axis=0)

        it += 1
        print("iteration {} complete".format(it))


    return mu


if __name__ == '__main__':

    A = imread('peppers-large.tiff')
    plt.imshow(A)
    plt.show()

    #print(A.shape)

    #Get small image
    B = imread('peppers-small.tiff')
    plt.imshow(B)
    plt.show()

    #reshape small image pixels into one column ( with three rows, for r,g,b)
    m, n, r = B.shape
    points = B.reshape((m*n, r))
    #Find the mean values of the 16 color values
    mu = k_means(points, 16)

    #Reshape A pixels into one column
    m, n, r = A.shape
    A_new = A.reshape((m*n, r))
    print(A_new.shape)

    #Find c for A (closest color centroid values)
    w = np.linalg.norm(A_new[:, None, :] - mu[None, :, :], axis=2)
    # c (m,)
    c = np.argmin(w, axis=1)

    #Change each pixel to value of closest color centroid
    A_new = mu[c].astype(np.uint8)

    #reshape A into 3D matrix
    A_new = A_new.reshape(m,n,r)
    #print(A_new)

    plt.imshow(A_new)
    plt.show()
