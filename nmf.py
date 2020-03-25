import numpy as np
import random
from data import *
from evaluation import *

class nmf_sgd():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 learning_rate=0.001,   # learning_rate: the learning rata
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
    
    
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))

        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        records_list = []
        for step in range(self.max_iteration):
            los=0.0
            for data in self.train_list:
                u,i,r = data
                P[u],Q[i],ls = self.update(P[u], Q[i], r=r, learning_rate=self.learning_rate)
                los += ls
            pred_mat = self.prediction(P,Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P,Q,np.array(records_list)


    def update(self, p, q, r, learning_rate=0.001):
        error = r - np.dot(p, q.T)            
        p = p + learning_rate*error*q
        q = q + learning_rate*error*p
        loss = 0.5 * error**2 
        return p, q, loss


    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
		
		
class nmf_mult():
    def __init__(self, 
                 train_list,            # train_list: train data 
                 test_list,             # test_list: test data
                 N,                     # N:the number of user
                 M,                     # M:the number of item
                 K=10,                  # K: the number of latent factor
                 max_iteration=50       # max_iteration: the max iteration
                ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.max_iteration = max_iteration
    
    
    def train(self):
        train_mat = sequence2mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence2mat(sequence = self.test_list, N = self.N, M = self.M)

        avg = np.sqrt(train_mat.mean() / self.K)
        P = avg*np.random.normal(0, 1., (self.N, self.K))
        Q = avg*np.random.normal(0, 1., (self.M, self.K))

        records_list = []
        for step in range(self.max_iteration):
            P,Q = self.update(P, Q, R=train_mat)
            user = np.array(self.train_list)[:,0].astype(np.int16)
            item = np.array(self.train_list)[:,1].astype(np.int16)
            rating_true = np.array(self.train_list)[:,2]
            rating_pred = np.sum(P[user,:]*Q[item,:],axis=1)
            los = np.sum((rating_true-rating_pred)**2)
            pred_mat = self.prediction(P,Q)
            mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
            records_list.append(np.array([los, mae, rmse, recall, precision]))

            if step % 10 ==0:
                print(' step:%d \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
                      %(step,los,mae,rmse,recall,precision))

        print(' end. \n loss:%.4f,mae:%.4f,rmse:%.4f,recall:%.4f,precision:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3],records_list[-1][4]))
        return P,Q,np.array(records_list)


    def update(self, P, Q, R ,eps=1e-6):            
        P = P * (np.dot(R+eps,Q)/(np.dot(P,np.dot(Q.T,Q)))+eps)
        Q = Q * (np.dot(R.T+eps,P)/(np.dot(Q,np.dot(P.T,P)))+eps)
        return P, Q
    
    
    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred