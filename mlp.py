import tensorflow as tf
import numpy as np
from data import *
from evaluation import *

class mlp():
    def __init__(self,               
                 users_num = None, #用户数
                 items_num = None, #商品数
                 embedding_size = 16, # 嵌入空间维度
                 hidden_sizes = [16,8], #隐层节点数目
                 learning_rate = 1e-3, #学习率
                 lamda_regularizer=1e-3, #正则项系数
                 batch_size = 256 #batch大小
                ):
        self.users_num = users_num
        self.items_num = items_num
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.batch_size = batch_size

        # loss records
        self.train_loss_records = []  
        self.build_graph()   

        
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():       
            # _________ input data _________
            self.users_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None], name='users_inputs')
            self.items_inputs = tf.compat.v1.placeholder(tf.int32, shape = [None], name='items_inputs')
            self.train_labels = tf.compat.v1.placeholder(tf.float32, shape = [None], name='train_labels') 
            
            # _________ variables _________
            self.weights = self._initialize_weights()
            
            # _________ train _____________
            self.y_ = self.inference(users_inputs=self.users_inputs, items_inputs=self.items_inputs)
            self.loss_train = self.loss_function(true_labels=self.train_labels, 
                                                 predicted_labels=tf.reshape(self.y_,shape=[-1]),
                                                 lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 

            # _________ prediction _____________
            self.predictions = self.inference(users_inputs=self.users_inputs, items_inputs=self.items_inputs)
        
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

        # -----embedding layer------
        all_weights['embedding_users'] = tf.Variable(tf.random.normal([self.users_num, self.embedding_size],0, 0.1),name='embedding_users')
        all_weights['embedding_items'] = tf.Variable(tf.random.normal([self.items_num, self.embedding_size],
                                                                      0, 0.1),name='embedding_items') 
        
        # ------hidden layer------
        all_weights['weight_0'] = tf.Variable(tf.random.normal([self.embedding_size * 2,self.hidden_sizes[0]], 0.0, 0.1),name='weight_0')
        all_weights['bias_0'] = tf.Variable(tf.zeros([self.hidden_sizes[0]]), name='bias_0')
        all_weights['weight_1'] = tf.Variable(tf.random.normal([self.hidden_sizes[0],self.hidden_sizes[1]], 0.0, 0.1), name='weight_1')
        all_weights['bias_1'] = tf.Variable(tf.zeros([self.hidden_sizes[1]]), name='bias_1')
        
        # ------output layer-----
        all_weights['weight_n'] = tf.Variable(tf.random.normal([self.hidden_sizes[-1], 1], 0, 0.1), name='weight_n')
        all_weights['bias_n'] = tf.Variable(tf.zeros([1]), name='bias_n')

        return all_weights
        
    
    def train(self, data_sequence):
        train_size = len(data_sequence)
        batch_size = self.batch_size
        total_batch = math.ceil(train_size/batch_size)

        for batch in range(total_batch):
            start = (batch*batch_size)% train_size
            end = min(start+batch_size, train_size)
            data_array = np.array(data_sequence[start:end])
            X = data_array[:,:2] # u,i
            y = data_array[:,-1] # label

            feed_dict = {self.users_inputs: X[:,0], self.items_inputs: X[:,1], self.train_labels:y}  
            loss, opt = self.sess.run([self.loss_train,self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            
        return self.train_loss_records

        
    # 网络的前向传播
    def inference(self, users_inputs, items_inputs):
        embed_users = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_users'], users_inputs),
                                 shape=[-1, self.embedding_size])
        embed_items = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_items'], items_inputs),
                                 shape=[-1, self.embedding_size])
            
        layer0 = tf.nn.relu(tf.matmul(tf.concat([embed_items,embed_users],1), self.weights['weight_0']) + self.weights['bias_0'])
        layer1 = tf.nn.relu(tf.matmul(layer0, self.weights['weight_1']) + self.weights['bias_1'])       
        y_ = tf.matmul(layer1,self.weights['weight_n']) + self.weights['bias_n']
        return y_         
        
        
    def loss_function(self, true_labels, predicted_labels,lamda_regularizer=1e-3):   
        loss = tf.compat.v1.losses.mean_squared_error(true_labels, predicted_labels)
        cost = loss
        if lamda_regularizer>0:
            regularizer_1 = tf.contrib.layers.l2_regularizer(lamda_regularizer)
            regularization = regularizer_1(
                self.weights['embedding_users']) + regularizer_1(
                self.weights['embedding_items'])+ regularizer_1(
                self.weights['weight_0']) + regularizer_1(
                self.weights['weight_1']) + regularizer_1(
                self.weights['weight_n'])
            cost = loss + regularization

        return cost   
    
    
    def predict_ratings(self, data_sequence):
        pred_mat = np.zeros([self.users_num, self.items_num])
        
        instances_size = len(data_sequence)
        data_array = np.array(data_sequence)
        items_id = np.array([i for i in range(self.items_num)])
        for u in range(self.users_num):
            users_id = u*np.ones_like(items_id)
            feed_dict = {self.users_inputs:users_id, 
                         self.items_inputs:items_id}  
            out = self.sess.run([self.predictions], feed_dict=feed_dict)
            pred_mat[u] = np.reshape(out,(-1))

        return pred_mat
