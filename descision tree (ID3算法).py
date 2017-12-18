# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:18:52 2017

@author: Zzz~L
"""
""" 决策树算法实现流程：计算信息熵——划分数据集——找到最好特征划分数据集（该特征在划分数据集后，信息增益最大）——
    利用递归函数构建决策树函数（对于将所有属性都用完了，但仍然存在不唯一标签的分支，采用投票表决的方式决定该分支的类标签）
    key=operator.itemgetter(1)表示按第一个参数排序
"""
#------------------------------机器学习实战第三章决策树----------------------
#=========================计算熵=========================
from math import log
import operator
def calcShannonEnt(dataSet):
    m=len(dataSet)
    label={}
    for ds in dataSet:
        lab=ds[-1]
        label[lab]=label.get(lab,0)+1#当lab in label,则返回lab的key
    Ent=0
    for key in label:
        prob=float(label[key])/m
        Ent -= prob*log(prob,2) #计算信息熵
    return Ent
def createdataset():
    dataSet=[[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
#===========按照给定特征划分数据集,并返回剔除该特征后的数据=============
def splitDataSet(dataSet, axis, value):#axis指特征,value表示该特征取值
    newdataset=[]
    for ds in dataSet:
        if ds[axis]==value:#若该观测满足特征为指定数
            reducefeat=ds[:axis]#除去该特征的值,余下的特征
            reducefeat.extend(ds[axis+1:])
            newdataset.append(reducefeat)#满足条件的观测余下的特征
    return newdataset
#======================选择划分数据集的最好特征========================
def chooseBestFeatureToSplit(dataSet):
    baseEntropy = calcShannonEnt(dataSet)#计算原始信息熵
    k=len(dataSet[0])-1#特征个数
    bestInfoGain=0
    bestfeature=-1 ##为什么要先赋值
    for i in range(k):
        feat=[sample[i] for sample in dataSet]#第i个特征所有样本取值
        uniqval=set(feat)
        newEntropy=0 #每个特征都有新的信息熵
        for val in uniqval:#对于每种取值
            subsetdata=splitDataSet(dataSet,i,val)#划分数据集
            prob=len(subsetdata)/len(dataSet)#该取值下分类数占比
            newEntropy += prob*calcShannonEnt(subsetdata)#计算新的信息熵,总信息熵是子集信息熵的加权平均值(期望) H=3/5H(A)+2/5H(B)
        infoGain = baseEntropy - newEntropy  
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestfeature=i
    return bestfeature ##返回第i个特征是最好特征
#======================构建决策树=============================
""" 当使用完所有特征,但仍有分支下不是唯一的类标签,此时通常采用投票
   表决法决定该叶子节点的分类
""" 
##投票表决
def majorityCnt(classList):
    classcount={}
    for clas in classList:
        classcount[clas]=classcount.get(clas,0)+1
    sortedclass=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)#classcount.items以列表格式获得参数值,operator.itemgetter(1)表示按第一个参数排序
    return sortedclass[0][0]
#采用递归函数构建决策树（核心函数）
def createTree(dataSet,labels):
    classList=[sample[-1] for sample in dataSet]
    if len(set(classList))==1:#若该节点下只有一类标签
        return classList[0]
    if len(dataSet[0])==1:##若所有属性都用完
        return majorityCnt(classList)
    bestfeat=chooseBestFeatureToSplit(dataSet)#寻找最好特征
    bestlabel=labels[bestfeat]
    mytree={bestlabel:{}}#构建树字典
    del(labels[bestfeat])#剔除已使用过的标签
    featvalue=[sample[bestfeat] for sample in dataSet] #样本在最好特征上的取值
    uniqueval=set(featvalue)
    for val in uniqueval:##遍历所有属性值
        sublabels=labels[:]#在每个划分的数据集上继续递归调用函数,并将返回的值插入字典变量中
        mytree[bestlabel][val]=createTree(splitDataSet(dataSet,bestfeat,val),sublabels)
    return mytree
#分类函数（递归）构建决策树后,在测试集上实现决策树分类
def classify(inputTree,featLabels,testVec):
    firstfeat=list(inputTree.keys())[0]
    seconddic=inputTree[firstfeat]
    featindex=featLabels.index(firstfeat)
    testval=testVec[featindex]#测试集在该特征上的取值
    secondval=seconddic[testval]#该特征取值所对应的树分支
    if isinstance(secondval,dict):#是字典,则继续用其他特征划分
        classlabel=classify(secondval,featLabels,testVec)
    else:
        classlabel=secondval
    return classlabel
##隐形眼镜示例
import pandas as pd
from numpy import *
lense=pd.read_table('machinelearninginaction/Ch03/lenses.txt',index_col=None,header=None,names=('age','prescript','astigmatic','tearrate','class'))
dataSet=array(lense.iloc[:,:5]).tolist()#必须将原始数据转换为列表格式 tolist将数组转换为列表
labels=['age','prescript','astigmatic','tearrate']
lensetree=createTree(dataSet,labels)   
    
    



































