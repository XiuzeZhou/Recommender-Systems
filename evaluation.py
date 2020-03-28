import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pylab import *
import matplotlib
import matplotlib.pyplot as plt

def get_topn(r_pred, train_mat, n=10):
    unrated_items = r_pred * (train_mat==0)
    idx = np.argsort(-unrated_items)
    return idx[:,:n]


def recall_precision(topn, test_mat):
    n,m = test_mat.shape
    hits,total_pred,total_true = 0.,0.,0.
    for u in range(n):
        hits += len([i for i in topn[u,:] if test_mat[u,i]>0])
        size_pred = len(topn[u,:])
        size_true = np.sum(test_mat[u,:]>0,axis=0)
        total_pred += size_pred
        total_true += size_true

    recall = hits/total_true
    precision = hits/total_pred
    return recall, precision	
	
	
def mae_rmse(r_pred, test_mat):
    y_pred = r_pred[test_mat>0]
    y_true = test_mat[test_mat>0]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse 


def evaluation(pred_mat, train_mat, test_mat):
    topn = get_topn(pred_mat, train_mat, n=10)
    mae, rmse = mae_rmse(pred_mat, test_mat)
    recall, precision = recall_precision(topn, test_mat)
    return mae, rmse, recall, precision
	
	
def figure(values_list, name=''):
    fig=plt.figure(name)
    x = range(len(values_list))
    plot(x, values_list, color='g',linewidth=3)
    plt.title(name + ' curve')
    plt.xlabel('Iterations')
    plt.ylabel(name)
    show()
