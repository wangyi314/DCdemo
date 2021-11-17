#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class DCPredictor:
    def __init__(self,path,n):
        tf.reset_default_graph()
        self.dataFile = path        #数据文件路径
        self.n = n                  #当前进行训练或预测的是第几号球

        # lstm 的各个参数
        self.timeStep = 10
        self.hiddenUnitSize = 20  # 隐藏层神经元数量
        self.batchSize = 20  # 每一批次训练多少个样例
        self.inputSize = 1  # 输入维度
        self.outputSize = 1  # 输出维度
        self.lr = 0.0006  # 学习率
        self.train_x, self.train_y = [], []  # 训练数据集
        self.sortedChargeList = []  # 排序的训练数据集
        self.normalizeData = []  # 归一化的数据
        self.X = tf.placeholder(tf.float32, [None, self.timeStep, self.inputSize])
        self.Y = tf.placeholder(tf.float32, [None, self.timeStep, self.inputSize])
        self.weights = {
            'in': tf.Variable(tf.random_normal([self.inputSize, self.hiddenUnitSize])),
            'out': tf.Variable(tf.random_normal([self.hiddenUnitSize, 1]))
        }

        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.hiddenUnitSize, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

    #原始数据读取
    def loadData(self):
        fp = open(self.dataFile, 'rb')

        self.date_l = []            #存放日期
        self.num_l = []             #存放中奖号码
        #从文件种读取所需的数据
        while True:
            line = fp.readline().decode()
            if not line:
                break
            data = line.split(";")
            if len(data) < 2:
                continue
            num = re.findall(r'\d+',data[2])        #从(2019-07-21;2019084;04,08,14,18,20,27,03)提取出[04,08,14,18,20,27,03]
            num = num[self.n]                       #从[04,08,14,18,20,27,03]中选择第n个
            self.num_l.append(int(num))

    # 构造满足LSTM的训练数据
    def buildTrainDataSet(self):
        self.num_l.reverse()
        self.meanNum = np.mean(self.num_l)      #平均值
        self.stdNum = np.std(self.num_l)        #标准差
        self.Data = (self.num_l - self.meanNum) / self.stdNum  # 标准化

        self.Data = self.Data[:, np.newaxis]  # 增加维度
        for i in range(len(self.Data)-self.timeStep-1):
            x = self.Data[i:i+self.timeStep]
            y = self.Data[i+1:i+self.timeStep+1]
            self.train_x.append(x)
            self.train_y.append(y)

    # lstm算法定义
    def lstm(self, batchSize = None):
        if batchSize is None :
            batchSize = self.batchSize
        weightIn = self.weights['in']
        biasesIn = self.biases['in']
        input = tf.reshape(self.X, [-1,self.inputSize])
        inputRnn=tf.matmul(input,weightIn)+biasesIn
        inputRnn=tf.reshape(inputRnn,[-1,self.timeStep,self.hiddenUnitSize])  #将tensor转成3维，作为lstm cell的输入
        cell=tf.nn.rnn_cell.BasicLSTMCell(self.hiddenUnitSize)
        initState=cell.zero_state(batchSize,dtype=tf.float32)
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, inputRnn,initial_state=initState, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output=tf.reshape(output_rnn,[-1,self.hiddenUnitSize]) #作为输出层的输入
        w_out=self.weights['out']
        b_out=self.biases['out']
        pred=tf.matmul(output,w_out)+b_out
        return pred,final_states

    # 训练模型
    def trainLstm(self):
        print('begin to train NO:'+str(self.n+1))
        pred,_ = self.lstm()
        #定义损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(self.Y, [-1])))
        #定义训练模型
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 重复训练100次，训练是一个耗时的过程
            for i in range(100):
                step = 0
                start = 0
                end = start + self.batchSize
                while end < len(self.train_x):
                    _, loss_ = sess.run([train_op, loss], feed_dict={self.X: self.train_x[start:end], self.Y: self.train_y[start:end]})
                    start += self.batchSize
                    end = start + self.batchSize
                    if step % 20 == 0:
                        print('num:%d  step:%d  loss:%f'%(i,step,loss_))
                    step += 1
            #训练完成保存模型
            saver.save(sess, './DCModel_'+str(self.n)+'/stock.model')

    #进行预测
    def prediction(self):
        print('begin to prediction NO:'+str(self.n+1))
        pred, _ = self.lstm(1)  # 预测时只输入[1,time_step,inputSize]的测试数据
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint('./DCModel_'+str(self.n))
            saver.restore(sess, module_file)
            # 取训练集最后一行为测试样本. shape=[1,time_step,inputSize]
            prev_seq = self.train_y[-1]
            next_seq = sess.run(pred, feed_dict={self.X: [prev_seq]})
            return int(next_seq[-1][0]*self.stdNum+self.meanNum+0.5)


if __name__ == '__main__':
    path = '.\DataPreparation\DCnumber.txt'
    type = 1             #0表示训练，1表示预测
    preNumber = []      #存放预测出来的值
    for n in range(7):    #0-5表示从1到6个红球，6表示篮球
        predictor = DCPredictor(path,n)
        predictor.loadData()
        # 构建训练数据
        predictor.buildTrainDataSet()
        if type == 0:
            # 模型训练
            predictor.trainLstm()
        else:
            # 预测－预测前需要先完成模型训练
            number = predictor.prediction()
            preNumber.append(number)
    print(preNumber)
