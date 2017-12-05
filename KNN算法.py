
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:26:34 2017
@author: Zzz~L
"""
"""KNN:预测新样本所属类别，利用距离样本最近的k个已知类别的样本，选择k个样本中出现次数
       最多的类别
       知识小结：1.tile(y,(4,1)) tile对y产生4*1维数组 将变量内容复制成输入矩阵同样大小的矩阵
       2.distance.argsort() 对distance排序,并返回由小到大的值的index
       3.count.get(lab,0)+1 count.get表示若lab in count,则返回lab的key,反之返回0
       4.label=dataset['label'].copy() 选择复制后,再修改label时,不会改动原变量
       5.label[label[:]=='didntLike']=1 此方法重新编码才不会产生SettingWithCopy
       6.利用循环方式完成分组散点图 第61行
       7.set_printoptions(threshold=Inf) 设置数组显示数目,Inf表示显示边界无穷
"""
#--------------------------机器学习实战第二章KNN算法------------------------------
import numpy as np
matr=np.mat(np.random.rand(4,4))#内部随机产生4*4数组,外部转化为矩阵
#===========================KNN算法概述===========================
from numpy import *
def classify0(inX, dataSet, labels, k):
    size=dataSet.shape[0]#返回观测值个数
    distance=((tile(inX,(size,1))-dataSet)**2).sum(axis=1)**0.5##计算欧式距离,tile对y产生4*1维数组,axi=1表示按行求和
    sort=distance.argsort()#对distance排序,并返回由小到大的值的index
    count={}
    for i in range(k):
        lab=labels[sort[i]]
        count[lab]=count.get(lab,0)+1#count.get表示若lab in count,则返回lab的key,反之返回0
    sortlist=sorted(count.items(),reverse=True)#count.items()返回字典中的项目
    return sortlist[0][0]
#===========================KNN算法示例============================
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
filename='datingTestSet.txt'
def file2matrix(filename):
    dataset=pd.read_table(filename,header=None,names=('plane','game','ice','label'))
    datamatrix=dataset.iloc[:,(0,1,2)]
    label=dataset['label'].copy()#选择复制后,再修改label时,不会改动原变量
    label[label[:]=='didntLike']=1#此方法重新编码才不会产生SettingWithCopy
    label[label[:]=='smallDoses']=2
    label[label[:]=='largeDoses']=3  
    label=label.astype('int')
    return datamatrix,label
###游戏时间与冰淇淋
fig,ax=plt.subplots(1,1)
ax.scatter(datamatrix.iloc[:,1],datamatrix.iloc[:,2], 
           s=15.0*array(label),c=15.0*array(label))#s表示大小差异,c表示颜色差异
ax.set_xlabel('play game time percent')
ax.set_ylabel('the number of ice')
####飞行里程数与游戏时间
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(datamatrix.iloc[:,0],datamatrix.iloc[:,1],s=15.0*array(label),c=15.0*array(label),label=array(label),edgecolors='none')
ax1.set_xlabel('plane time')
ax1.set_ylabel('the number of ice')
##含有图例的飞行里程数与游戏时间
colors = ['steelblue', '#9999ff', '#ff9999']
labels=list(set(dataset['label']))
fig, ax = plt.subplots()##利用循环方式完成分组散点图
for i in range(len(labels)):
    ax.scatter(dataset.loc[dataset.label==labels[i],'plane'],
               dataset.loc[dataset.label==labels[i],'game'],
               c=colors[i],label=labels[i])
ax.legend()
ax.set_xlabel('plane time')
ax.set_ylabel('the number of ice')
plt.show()
##变量归一化 newvalue=(oldvalue-min)/(max-min)
def autoNorm(datamatrix):
    Min=datamatrix.min(0)#0表示返回每列的最小值,1表示返回每行最小值
    Max=datamatrix.max(0)
    minmatrix=tile(Min,(1000,1))#tile()函数将变量内容复制成输入矩阵同样大小的矩阵
    maxmatrix=tile(Max,(1000,1))
    normdata=(datamatrix-minmatrix)/(maxmatrix-minmatrix)
    return normdata
##分类器对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datamatrix,label=file2matrix(filename)
    normdata=autoNorm(datamatrix)
    normdata=array(normdata)
    label=array(label)#label需转换为数组,否则后续取数报错
    test=int(normdata.shape[0]*hoRatio)
    errorcount=0
    for i in range(test):
        result=classify0(normdata[i,:], normdata[test:,:],label[test:],3)
        print("the classifier came back with:{},the real answer is:{}".format(result, label[i]))
        if result!=label[i]:
            errorcount +=1
    print("the total error rate is:{}".format(errorcount/test))
    print(errorcount)
#===========================KNN识别手写数字============================            
from numpy import *  
import pandas as pd
from os import listdir ##导入listdir函数,列出给定目录的文件名
set_printoptions(threshold=Inf)##设置数组显示数目,Inf表示显示边界无穷
##将图像转化为向量格式
def img2vector(filename):
    returnvector=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linestr=fr.readline()#按行读取图像矩阵，每次读取一行并以字符串的形式存储在lineStr中
        for j in range(32):
            returnvector[0,32*i+j]=int(linestr[j])#利用行列关系逐个取出当前行的每一个字符，并转化为数字，构造出1*1024的向量
    return returnvector  
def handwritingClassTest():
    trainfile=listdir('digits/trainingDigits')
    testfile=listdir('digits/testDigits')
    trainlabel=[int(i[0]) for i in trainfile]#从文件名中提取每个train文件的label
    testlabel=[int(j[0]) for j in testfile]
    m=len(trainfile)
    n=len(testfile)
    trainmat=zeros((m,1024))
    testmat=zeros((n,1024))
    errorcount=0
    for i in range(m):#转换训练集图像
        trainmat[i,:]=img2vector('digits/trainingDigits/{}'.format(trainfile[i]))
    for j in range(n):
        test=img2vector('digits/testDigits/{}'.format(testfile[j]))#转换测试集图像
        result=classify0(test,trainmat,trainlabel, 3)#KNN预测结果
        print("the classifier came back with:{},the real answer is:{}".format(result, testlabel[j]))
        if result != testlabel[j]:
            errorcount +=1
    print("the total error rate is:{}".format(errorcount/n))
    print("he total number of errors:{}".format(errorcount))
        








