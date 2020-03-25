import numpy as np
import scipy
from sklearn.utils.extmath import randomized_svd


def puresvd(R = None, # train mat
            k=150, # the number of latent factor
            ):
    P, sigma, QT = randomized_svd(R, k)
    sigma = scipy.sparse.diags(sigma, 0)
    P = P * sigma
    Q = QT.T   
    # R_= np.dot(P, QT)
    R_ = np.dot(R, np.dot(Q, QT)) #
    return R_