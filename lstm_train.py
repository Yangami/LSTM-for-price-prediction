import tensorflow as tf
import numpy as np
import pandas as pd
import time
import random
import csv
import os
import re
import shutil
import os
import configparser
#读配置文件获取读数据地址和模型保存地址
def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    train = cf.items("train")
    dic = {}
    for key, val in train:
        dic[key] = val
    batchsize =int(dic['batchsize'])
    epoch = int(dic['epoch'])
    timestep = int(dic['timestep'])
    trainpath = dic['path']
    trainbg=int(dic['trainbg'])
    trained=int(dic['trained'])
    trainlog=dic['trainlog']
    logtitle=dic['train_log_title']
    modelpath=dic['modelpath']
    columns=dic['columns']
    train = cf.items("valid")
    dic1 = {}
    for key, val in train:
        dic1[key] = val
    valbg=int(dic1['valbg'])
    valed=int(dic1['valed'])
    return batchsize,epoch,timestep,trainpath,trainbg,trained,valbg,valed,trainlog,logtitle,modelpath,columns
#读取数据
batchsize,epoch,timestep,trainpath,trainbg,trained,valbg,valed,trainlog,logtitle,modelpath,columns=parse_args('cfg.txt')
df=pd.read_csv(open(trainpath),index_col=0)
df.columns=['r']+columns.split(',')
df=df.iloc[:,:].dropna(axis=0)
print('数据shape：',np.shape(df))
#df['volume']=pd.Series([float(i) for i in df['volume']],index=df.index)
#df['volum_index']=pd.Series([float(i) for i in df['volum_index']],index=df.index)
seeds=123
data=df
#超参数初始化     #hidden layer units
input_size=len(df.columns)-1
print('输入维度',input_size)
output_size=2
lr=0.15
decay_rate = 0.93    #衰减系数
global_steps = 1000   #迭代次数
decay_steps = 100  #衰减步数
keep_prob=1
number_layers=2

#获取训练数据集
def get_train_data(data,batch_size,time_step,train_begin,train_end):#g.
    batch_index=[]
    data_train=data.iloc[train_begin:train_end].values
    normalized_train_data=(data_train[:,1:input_size+1]-np.mean(data_train[:,1:input_size+1],axis=0))/np.std(data_train[:,1:input_size+1],axis=0)#标准化，标准差能否提前计算？

    train_x,train_y=[],[]#训练集
    for i in range(len(normalized_train_data)-time_step):
        if i%batch_size==0:
            batch_index.append(i)
        x=normalized_train_data[i:i+time_step,:]
        y=data_train[i+time_step-1,0,np.newaxis]#一个时间步中最后一个
        train_x.append(x.tolist())
        train_y.extend(y.tolist())
    random.seed(seeds)
    random.shuffle(train_x)
    random.seed(seeds)
    random.shuffle(train_y)
    train_onehot_y=[]
    for i in train_y:
        i=(i//1)+2
        train_onehot_y.append([0]*int(i-1)+[1]+[0]*int(2-i))
    #for i in range(len(normalized_train_data)-time_step):
     #   train_onehot_y=train_onehot_y[i:i+time_step]
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_onehot_y
#验证数据
def get_test_data(data,time_step,test_begin,test_end):
    data_test=data.iloc[test_begin:test_end].values
    normalized_train_data=(data_test[:,1:input_size+1]-np.mean(data_test[:,1:input_size+1],axis=0))/np.std(data_test[:,1:input_size+1],axis=0)#标准化，标准差能否提前计算？
    test_x,test_y=[],[]#训练集
    for i in range(len(normalized_train_data)-time_step):
        x=normalized_train_data[i:i+time_step,:]
        y=data_test[i+time_step-1,0,np.newaxis]#一个时间步中最后一个
        test_x.append(x.tolist())
        test_y.extend(y.tolist())
    mean=np.mean(data_test[:,1:input_size+1],axis=0)
    std=np.std(data_test[:,1:input_size+1],axis=0)
    train_onehot_y=[]
    for i in test_y:
        i=(i//1)+2
        train_onehot_y.append([0]*int(i-1)+[1]+[0]*int(2-i))
#    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
#    test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
    return mean,std,test_x,train_onehot_y

#data=get_data(e+'600004'+'.csv')
#get_train_data(data,60,20,1,-1)


def lstm(X):
    batch_size=tf.shape(X)[0]#shape
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input = tf.nn.dropout(input, keep_prob)
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell= tf.nn.rnn_cell.MultiRNNCell([cell] *number_layers,state_is_tuple=True)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    cell_dr = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=0.5)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell_dr, input_rnn,initial_state=init_state,dtype=tf.float32)
   #print('out_putshape',np.shape(output_rnn),np.shape(output))
    output_rnn=tf.reshape(output_rnn,[time_step,rnn_unit,-1]) #作为输出层的输入
    #output_rnn = tf.transpose(output_rnn, [1,0,2])
    #final_states=tf.reshape(final_states,[-1,rnn_unit])
    w_out=weights['out']
    b_out=biases['out']
    tempState1=[tf.matmul(tf.transpose(output_rnn[-1],[1,0]),w_out)+b_out]
    tempState1 = tf.nn.relu(tempState1)
    #tempState2=tf.nn.dropout(tempState1,keep_prob=0.5)
    tempState2 = [tf.matmul(tempState1[0], weights['fc2']) + biases['fc2']]
    tempState2 = tf.nn.relu(tempState2)
    logits = tf.matmul(tempState2[0], weights['logit']) + biases['logit']
    pred=tf.nn.softmax(logits)
    return pred,logits,final_states

def train_lstm(data,epochs,batch_size,time_step,train_begin,train_end,valbg,valed):
    X=tf.placeholder(tf.float32,name='X', shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, name='Y',shape=[None,output_size])
    batch_index,train_x,train_y=get_train_data(data,batch_size,time_step,train_begin,train_end)
    pred,logits,final_states=lstm(X)
    #损失函数
    rnn_labels=[]
    #for i in range(Y.shape[0]):
     #   print(i)
      #  rnn_labels.append(tf.one_hot(Y[i], depth=2))
    rnn_labels=Y
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits)
    loss = tf.reduce_mean(losses)
    #loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    global_ = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(lr, global_, decay_steps, decay_rate, staircase=True)
    train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #saver.restore(sess, module_file)
        #重复训练10000次
        best_acclst=[0]*4#best_acc初始化
        flag=0
        # if os.path.exists('C:\\Users\\yy\\Desktop\\神经网络\\50周线收益率预测\\lstm周线模型\\result\\loss&acc.txt'+str(rnn_unit)+';0.96.txt'):
        #     os.remove('C:\\Users\\yy\\Desktop\\神经网络\\50周线收益率预测\\lstm周线模型\\result\\loss&acc.txt'+str(rnn_unit)+';0.96.txt')
        with open(trainlog, 'a') as acc_file:
            acc_file.write(logtitle)
            for epoch in range(epochs):#参数化
                #epoch=i//500
                for step in range(len(batch_index)-1):

                    _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                #print(epoch,loss_)
                if epoch % 20==0:
                    print("保存模型：",rnn_unit,saver.save(sess,modelpath+'st.model'))#'model/model'+str(i)+'/.ckpt'
                    #saver.save(sess, 'model1/stock.model')

                    acc = prediction(data, input_size, epoch, test_begain=valbg, test_end=valed, time_step=time_step)
                    for i,best_acc in enumerate(best_acclst):
                        if acc>=best_acc:
                            flag+=1
                            best_acclst[i] = acc
                            dir = modelpath
                            bestdir = './bestmodel'+str(i)
                            if os.path.exists(bestdir):
                                shutil.rmtree(bestdir)
                                shutil.copytree(dir, bestdir)
                                break
                            else:
                                shutil.copytree(dir, bestdir)
                                break
                        #saver.save(sess, 'model/model' + str(epoch + 1) + '/stock.ckpt')

                    acc_file.write('\n%s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
                    acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f' % (epoch, acc,loss_))

def prediction(data,input_size,epoch,test_begain,test_end,time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.int64,shape=[None,output_size])
    mean,std,test_x,test_y=get_test_data(data,time_step,test_begain,test_end)
    pred, logits, final_states=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint(modelpath)
        print(module_file)
        saver.restore(sess, module_file)
        test_predict=[]
        correctsample=0
        #for step in range(len(test_x)):
        pred=sess.run(pred,feed_dict={X:test_x})
        #prob=tf.nn.in_top_k(pred,test_y,1)
        pred_class_index = np.argmax(pred, 1)
        labels_l = np.argmax(test_y, 1)
        for i,j in enumerate(pred_class_index):
            if j ==labels_l[i]:
                correctsample+=1
        acc=correctsample/len(test_y)
        #predict=prob#.reshape((-1))
        #test_predict.extend(predict)
        test_y=np.array(test_y)
        #print('testpred:',pred_class_index,'y:',test_y,'len',len(pred_class_index))
        #acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/abs(test_y[:len(test_predict)])) #acc为测试集偏差
        #acc=correctsample/len(test_x)
        #print('acc=',acc)
        return acc
rnn_unit=56
n1,n2,n3=32,16,4
weights = {'in': tf.Variable(tf.random_normal([input_size, rnn_unit],seed = 123)),
           'out': tf.Variable(tf.random_normal([rnn_unit, n1],seed = 123)), 'fc2': tf.Variable(tf.random_normal([n1, n2],seed = 123456)),
           'fc3': tf.Variable(tf.random_normal([n2, n3],seed = 123)),
           'logit': tf.Variable(tf.random_normal([n2, 2],seed = 123))}
biases = {'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
          'out': tf.Variable(tf.constant(0.1, shape=[n1, ])), 'fc2': tf.Variable(tf.constant(0.1, shape=[n2, ])),
          'fc3': tf.Variable(tf.constant(0.1, shape=[n3, ])),
          'logit': tf.Variable(tf.constant(0.1, shape=[2]))}
with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
    train_lstm(data,epochs=epoch,batch_size=batchsize,time_step=timestep,train_begin=trainbg,train_end=trained,valbg=valbg,valed=valed)
   #     acc=prediction(data,input_size,test_begain=500,test_end=650,time_step=20)
#    #     print(acc)
    #best_acc=max(acc,best_acc)

