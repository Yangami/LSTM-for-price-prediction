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
import lstm_jqtest
result=pd.read_csv(open('./周线数据/test-zhendang.csv'),index_col=0)
result=result[['r','open','high','low','close','volume','rsrs','turnover','money','circulating_cap','MA5','MA10','MA20']]
data=result.dropna(axis=0)
#with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
pred= lstm_jqtest.prediction(data, test_begain=0, test_end=600)
print(pred)