import numpy as np
import random
from data import *
from evaluation import *

class wmf():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 alpha=40,              # alpha: the confidence of negtive samplers
                 lamda_regularizer=0.1, # lamda_regularizer: regularization parameters
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.alpha = alpha
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration
    
    
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        records_list = []
        for step in range(self.max_iteration):
            for u in range(self.N):
                Ru = train_mat[u,:]
                P[u,:] = self.update(Q, Ru, lamda_regularizer=self.lamda_regularizer, alpha=self.alpha)

            for i in range(self.M):
                Ri = train_mat[:,i]
                Q[i,:] = self.update(P, Ri.T, lamda_regularizer=self.lamda_regularizer, alpha=self.alpha)

            pred_mat = self.prediction(P, Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([mae, rmse, recall, precision]))

            print(' step:%d \n mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'%(step,mae,rmse,recall,precision))

        print(' end. \n mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3]))
        return P, Q, np.array(records_list)


    def update(self, P, Ru, lamda_regularizer=0.1, alpha=40):
        # P: N/M *K
        # Ru: N/M *1
        N, K = P.shape
        c_ui = 1 + alpha*Ru
        Cu = c_ui* np.eye(N)   

        YtCY_I = P.T.dot(Cu).dot(P) + lamda_regularizer*np.eye(K)
        YtCRu = P.T.dot(Cu).dot(Ru)
        p = np.linalg.inv(YtCY_I).dot(YtCRu)
        return p.T


    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred