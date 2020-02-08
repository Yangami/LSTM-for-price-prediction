import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import csv
import matplotlib.pyplot as plt
import os
import datetime
import statsmodels.api as sm
import time
import configparser
#读配置文件获取读数据地址和模型保存地址
def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    train = cf.items("test")
    dic = {}
    for key, val in train:
        dic[key] = val
    testdata=dic['testdata']
    testresult=dic['testresult']
    testmodel=dic['testmodel']


    return testdata,testresult,testmodel
testdata,testresult,testmodel=parse_args('cfg.txt')
since=time.time()
result=pd.read_csv(open(testdata),index_col=0)
result=result[['r','open','high','low','close','volume','rsrs','turnover','money','circulating_cap','MA5','MA10','MA20','MA60']]
data=result.dropna(axis=0)
print(len(data.index))
#参数和超参数初始化
rnn_unit=56      #hidden layer units
input_size=len(data.columns)-1
output_size=2
lr=0.15
decay_rate = 0.93    #衰减系数
global_steps = 300   #迭代次数
decay_steps = 100  #衰减步数
epoch=20
keep_prob=1
number_layers=2

def get_test_data(data,time_step,test_begin):
    data_test=data.iloc[test_begin:].values
    normalized_train_data=(data_test[:,1:input_size+1]-np.mean(data_test[:,1:input_size+1],axis=0))/np.std(data_test[:,1:input_size+1],axis=0)#标准化，标准差能否提前计算？
    test_x,test_y=[],[]#训练集
    for i in range(len(normalized_train_data)-time_step):
        x=normalized_train_data[i:i+time_step,:]
        y=data_test[i+time_step-1,0,np.newaxis]#一个时间步中最后一个
        test_x.append(x.tolist())
        test_y.extend(y.tolist())
    test_x.append(normalized_train_data[-time_step:,:].tolist())
    test_y.extend(data_test[-1,0,np.newaxis].tolist())
    mean=np.mean(data_test[:,1:input_size+1],axis=0)
    std=np.std(data_test[:,1:input_size+1],axis=0)
    train_onehot_y=[]
    for i in test_y:
        i=(i//1)+2
        train_onehot_y.append([0]*int(i-1)+[1]+[0]*int(2-i))
    return mean,std,test_x,train_onehot_y
n1,n2,n3=32,16,4
weights = {'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),#,seed = 123)),
           'out': tf.Variable(tf.random_normal([rnn_unit, n1],seed = 123)), 'fc2': tf.Variable(tf.random_normal([n1, n2],seed = 123)),
           'fc3': tf.Variable(tf.random_normal([n2, n3],seed = 123)),
           'logit': tf.Variable(tf.random_normal([n2, 2],seed = 123))}
#weights = {'in': tf.Variable(tf.constant(0.1,shape=[input_size, rnn_unit])),
 #          'out': tf.Variable(tf.constant(0.1,shape=[rnn_unit, n1])), 'fc2': tf.constant(0.1,shape=[n1, n2]),
  #         'fc3': tf.Variable(tf.constant(0.1,shape=[n2, n3])),
   #        'logit': tf.Variable(tf.constant(0.1,shape=[n2, 2]))}
biases = {'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
          'out': tf.Variable(tf.constant(0.1, shape=[n1, ])), 'fc2': tf.Variable(tf.constant(0.1, shape=[n2, ])),
          'fc3': tf.Variable(tf.constant(0.1, shape=[n3, ])),
          'logit': tf.Variable(tf.constant(0.1, shape=[2]))}
#checkpoint_dir = r'C:\Users\yy\PycharmProjects\untitled6\venv\stock2.model-680.index'
#checkpoint_dir=r'C:\Users\yy\PycharmProjects\untitled6\venv\stock2.model-680.meta'
#checkpoint_dir=r'C:\Users\yy\PycharmProjects\untitled6\venv\stock2.model-680.data-00000-of-00001'

def lstm(X):
    batch_size=tf.shape(X)[0]#shape
    time_step=tf.shape(X)[1]
    w_in=weights['in']#原因
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input = tf.nn.dropout(input, keep_prob)
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell= tf.nn.rnn_cell.MultiRNNCell([cell] *number_layers,state_is_tuple=True)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    cell_dr = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=0.6)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell_dr, input_rnn,initial_state=init_state,dtype=tf.float32)
   #print('out_putshape',np.shape(output_rnn),np.shape(output))
    output_rnn=tf.reshape(output_rnn,[time_step,rnn_unit,-1]) #作为输出层的输入
    #output_rnn = tf.transpose(output_rnn, [1,0,2])
    #final_states=tf.reshape(final_states,[-1,rnn_unit])
    w_out=weights['out']
    b_out=biases['out']
    tempState1=[tf.matmul(tf.transpose(output_rnn[-1],[1,0]),w_out)+b_out]#为什么是1
    tempState1 = tf.nn.relu(tempState1)
    #tempState2=tf.nn.dropout(tempState1,keep_prob=0.5)
    tempState2 = [tf.matmul(tempState1[0], weights['fc2']) + biases['fc2']]
    tempState2 = tf.nn.relu(tempState2)
    logits = tf.matmul(tempState2[0], weights['logit']) + biases['logit']
    pred=tf.nn.softmax(logits)
    return pred,logits,final_states

def prediction(data,input_size,epoch,test_begain,test_end,time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(data,time_step,test_begain)
    print(data['r'])
    pred, logits, final_states=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint(testmodel)
        saver.restore(sess, module_file)
        test_predict=[]
        correctsample=0
        #for step in range(len(test_x)):
        pred=sess.run(pred,feed_dict={X:test_x})
        print(test_y)
        pred_class_index = np.argmax(pred, 1)
        labels_l = np.argmax(test_y,1)
        for i,j in enumerate(pred_class_index):
            if j ==labels_l[i]:
                correctsample+=1
        acc=correctsample/len(test_y)
        check=pd.DataFrame({'pred':pred_class_index,'label':labels_l},index=list(data.index)[-len(pred):])
        check.to_csv(testresult)
        print('acc=',acc,'\n','prde:',pred_class_index,'\n','labl:',labels_l)
        return acc
with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
    acc = prediction(data, input_size, 2000, test_begain=0, test_end=600, time_step=10)
    print(time.time()-since)
