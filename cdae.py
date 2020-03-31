import tensorflow as tf
import numpy as np
from data import *
from evaluation import *


class cdae():
    def __init__(self,
                 users_num = None,         #用户数
                 items_num = None,         #商品数
                 hidden_size = 500,        #隐层节点数目，即用户的嵌入空间维度
                 batch_size = 256,         #batch大小
                 learning_rate = 1e-3,     #学习率
                 lamda_regularizer = 1e-3, #正则项系数
                 dropout_rate = 0.5        # dropout rate 
                ):
        self.users_num = users_num
        self.items_num = items_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.dropout_rate = dropout_rate
        
        self.train_loss_records = []  
        self.build_graph()   

        
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():      
            # _________ input data _________
            self.rating_inputs = tf.compat.v1.placeholder(tf.float32, shape = [None, self.items_num], name='rating_inputs')
            self.user_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None, 1], name='user_inputs')
            self.dropout_prob = tf.compat.v1.placeholder(tf.float32, name = "dropout_prob")

            # _________ variables _________
            self.weights = self._initialize_weights()
            
            # _________ train _____________
            self.y_ = self.inference(rating_inputs=self.rating_inputs, user_inputs=self.user_inputs)
            self.loss_train = self.loss_function(true_r=self.corrupted_inputs, predicted_r=self.y_, lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 
        
            # _________ prediction _____________
            self.predictions = self.inference(rating_inputs=self.rating_inputs, user_inputs=self.user_inputs)
            
            #变量初始化 init 
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
    
    
    def _init_session(self):
        # adaptively growing memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)
    
    
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['W1'] = tf.Variable(tf.random.normal([self.items_num, self.hidden_size], 0.0, 0.1), name='W1')
        all_weights['b1'] = tf.Variable(tf.zeros([self.hidden_size]), name='b1')
        all_weights['W2'] = tf.Variable(tf.random.normal([self.hidden_size, self.items_num], 0.0, 0.1), name='W2')
        all_weights['b2'] = tf.Variable(tf.zeros([self.items_num]), name='b2')
        all_weights['V'] = tf.Variable(tf.zeros([self.users_num, self.hidden_size]), name='V')
        return all_weights
    
    
    def train(self, data_mat):
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size)% instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.user_inputs:np.reshape(data_mat[start:end,0],(-1,1)), 
                         self.rating_inputs:data_mat[start:end,1:], 
                         self.dropout_prob:self.dropout_rate}  
            loss, opt = self.sess.run([self.loss_train, self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            
        return self.train_loss_records

        
    # 网络的前向传播
    def inference(self, rating_inputs, user_inputs):
        self.corrupted_inputs = tf.nn.dropout(rating_inputs, rate=self.dropout_prob)
        Vu = tf.reshape(tf.nn.embedding_lookup(self.weights['V'], user_inputs),(-1, self.hidden_size))
        encoder = tf.nn.sigmoid(tf.matmul(self.corrupted_inputs, self.weights['W1']) + Vu + self.weights['b1'])
        decoder = tf.identity(tf.matmul(encoder, self.weights['W2']) + self.weights['b2'])
        return decoder         
        
        
    def loss_function(self, true_r, predicted_r, lamda_regularizer=1e-3, loss_type='square'):
        if loss_type=='square':
            loss = tf.compat.v1.losses.mean_squared_error(true_r, predicted_r)
        elif loss_type=='cross_entropy':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_r, logits=predicted_r)
        
        regularizer = tf.contrib.layers.l2_regularizer(lamda_regularizer)
        regularization = regularizer(self.weights['V']) + regularizer(self.weights['W1']) + regularizer(
            self.weights['W2']) + regularizer(self.weights['b1']) + regularizer(self.weights['b2'])
        cost = loss + regularization
        return cost 
    
    
    def predict_ratings(self, data_mat):
        pred_mat = np.zeros([self.users_num, self.items_num])
        
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size)% instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.user_inputs:np.reshape(data_mat[start:end,0],(-1,1)), 
                         self.rating_inputs:data_mat[start:end,1:],
                         self.dropout_prob:0.}  
            out = self.sess.run([self.predictions], feed_dict=feed_dict)
            pred_mat[start:end,:] = np.reshape(out,(-1,self.items_num))

        return pred_mat