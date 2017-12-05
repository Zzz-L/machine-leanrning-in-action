# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:14:58 2017

@author: Zzz~L
"""
"""logistic回归寻找一个非线性函数Sigmoid的最佳拟合参数,通过最优化算法求解,梯度上升算法
   和随机梯度上升算法,后者是在线算法,在新数据到来时完成参数更新,不需要遍历整个数据集
   知识小结：1.datamat.insert(loc,'column',value)##在指定位置loc插入列column，其值为value
   2.mat(dataMatIn)转换为numpy数组
   3.随机梯度上升算法:步长会随着迭代次数逐渐减小,缓解数据波动;随机选择样本更新回归参数,减少周期性波动
   4.python格式化输出：打印字符串：print ("His name is %s"%("Aviad"))
   打印整数：print ("He is %d years old"%(25))  
   打印浮点数：print ("His height is %f m"%(1.83))
   打印浮点数(指定保留小数点位数)：print ("His height is %.2f m"%(1.83))
   指定占位符宽度：print ("Name:%10s Age:%8d Height:%8.2f"%("Aviad",25,1.83))
"""
#————————————————————————————————机器学习实战第五章logistic算法———————————————————————
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
testdata=pd.read_table('testSet.txt',header=None,names=('x1','x2','y'))
datamat=testdata.iloc[:,:2]
datamat.insert(0,'x0',1)##在指定位置插入列
labelmat=testdata['y']
#**********************主函数1***************
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
#梯度算法的迭代公式：w :=w+alpa*梯度  梯度=x.transpose()(y-xw)
def gradAscent(dataMatIn, classLabels):
    datamatrix=mat(dataMatIn)#转换为numpy矩阵,后续才能相乘
    labelmatrix=mat(classLabels).transpose()#转置
    m,n=datamatrix.shape
    alpha=0.001
    maxiter=500
    weight=ones((n,1))#初始化系数
    for k in range(maxiter):
        z=sigmoid(datamatrix*weight)#将得到的y值转换到0-1区间上
        gradient=datamatrix.transpose()*(labelmatrix-z)
        weight=weight+alpha*gradient
    return weight##回归系数
weight=gradAscent(datamat,labelmat)
#图形绘制 画出决策边界
def plotBestFit(weight,datamat,labelmat):
    datamat=array(datamat)
    xcord1=[];xcord2=[]
    ycord1=[];ycord2=[]
    for i in range(len(labelmat)):
        if labelmat[i]==1:
            xcord1.append(datamat[i,1])
            ycord1.append(datamat[i,2])
        else:
            xcord2.append(datamat[i,1])
            ycord2.append(datamat[i,2])
    fig,ax=plt.subplots(1)
    ax.scatter(xcord1,ycord1)
    ax.scatter(xcord2,ycord2)
    x = arange(-3.0, 3.0, 0.1)#生成给定区间以及间隔的数组
    y = (-weight[0]-weight[1]*x)/weight[2]# 0 = w0x0 + w1x1 +w2x2,解出x2和x1的关系式
    #y=array(y[0])
    ax.plot(x,y)
##改进版随机梯度上升算法(alpha逐渐减小,随机选择样本更新回归参数)
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n=dataMatrix.shape
    weight=ones(n)
    for j in range(numIter):
        dataindex=list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001 #alpha的取值？？？
            index=int(random.uniform(0,len(dataindex)))
            z=sigmoid(sum(dataMatrix[index]*weight))#需求和,计算w0x0+w1x1+w2x2
            gradient=dataMatrix[index].transpose()*(classLabels[index]-z)
            weight=weight+alpha*gradient
            del(dataindex[index])#需剔除已经使用过的样本
    return weight
#----------------------------示例 预测病马的死亡率--------------------
#给定测试集和回归系数,根据sigmoid函数转换
def classifyVector(inX, weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1
    else:
        return 0
#读取数据,训练模型,计算测试集错误率
def colicTest():
     frTrain = pd.read_table('horseColicTraining.txt',header=None)
     frTest = pd.read_table('horseColicTest.txt',header=None)
     trainmat=array(frTrain.iloc[:,:21])
     trainlabel=frTrain.iloc[:,21]
     testmat=array(frTest.iloc[:,:21])
     testlabel=frTest.iloc[:,21]
     weight=stocGradAscent1(trainmat,trainlabel,500)
     error=0
     for i in range(len(testmat)):
         clas=classifyVector(testmat[i], weight)
         if clas != testlabel[i]:
             error += 1
     errorate=float(error)/len(testmat)
     print("the error rate of this test is: %f" % errorate)
     return errorate
#重复函数colicTest()10次,计算错误率的平均值
def multiTest():
    count=10;errorsum=0
    for i in range(count):
        errorsum += colicTest()
    print("after %d iterations the average error rate is: %f" % (count, errorsum/float(count)))
    
    




        
