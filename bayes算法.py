# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:35:46 2017

@author: Zzz~L
"""
""" 朴素贝叶斯--根据概率进行分类,从已知类别的文档中,获得词条属于各类别的条件概率,再根据
    各词条的概率预测未知文档类别。
    知识小结：1.ones(k)-生成包含k个1的数组
    2.re.split(r'[^\w]',bigString)-将文本分隔为单个词汇,其中分隔符是除单词、数字外的任意字符串,[^ ]表示非字符集,[^\w]等价与\W
    3.'email/spam/%d.txt' % i等价于'email/spam/{}.txt'.format(i)
    4.划分测试集与训练集最基础的算法--第118行
    5.import feedparser ##RSS阅读器
"""
#-----------------------------第四章 朴素贝叶斯------------------------
from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
#********主函数1*********
##创建一个包含在所有文档中出现的不重复
def createVocabList(dataSet):
    vocaset=set([])#首先创建空集合
    for data in dataSet:
        vocaset=vocaset | set(data) #两个集合的并集
    return list(vocaset)
#********主函数2*********
##单词转化为词向量
def setOfWords2Vec(vocabList, inputSet):
    returnvec=[0]*len(vocabList)#生成指定长度的零向量
    for voca in vocabList:
        if voca in inputSet:
            index=vocabList.index(voca)
            returnvec[index]=1
        else:
            print("the word:{} is not in my Vocabulary!".format(voca))
    return returnvec
##构造文档矩阵
postingList,classVec=loadDataSet()
myvocab=createVocabList(postingList)
trainmat=[]#判断myvocab中的单词是否在postinglist中,构造词向量
for post in postingList:
    trainmat.append(setOfWords2Vec(myvocab,post))
#********主函数3*********
##计算词汇是侮辱性词汇的概率(训练算法)
def trainNB0(trainMatrix,trainCategory):
    sample=len(trainMatrix)
    number=len(trainMatrix[0])
    p1_percent=sum(trainCategory)/len(trainCategory)
    p0=ones(number);p1=ones(number)#所有词汇的初始数目
    p0sum=2;p1sum=2
    for i in range(sample):
        if trainCategory[i]==1:
            p1 += trainMatrix[i]#若该词条为侮辱性词条,则该词条中所有词汇加1
            p1sum += sum(trainMatrix[i])
        else:
            p0 +=trainMatrix[i]
            p0sum +=sum(trainMatrix[i])
    p1_prob=log(p1/p1sum)#计算每个词汇是侮辱性词汇的概率
    p0_prob=log(p0/p0sum)
    return p0_prob,p1_prob,p1_percent        
#********主函数4*********     
##计算文本是侮辱性文本的概率       
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #文本的总概率等于每个词汇的概率相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) #为什么要添加log(pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0  
#--------------------(示例一)使用朴素贝叶斯过滤网站恶意留言----------------
##封装所有函数
def testingNB():
    postingList,classVec=loadDataSet()
    myvocab=createVocabList(postingList)
    trainmat=[]#判断myvocab中的单词是否在postinglist中,构造词向量
    for post in postingList:
        trainmat.append(setOfWords2Vec(myvocab,post))
    p0v,p1v,pab=trainNB0(trainmat,classVec)  
    testEntry = ['love', 'my', 'dalmation']
    testvec=setOfWords2Vec(myvocab,testEntry)#首先将文档转换为词词向量
    class1=classifyNB(testvec, p0v,p1v,pab)#对文档分类
    print(testEntry,'classified as: ',class1)
    testEntry = ['stupid', 'garbage']
    testvec=setOfWords2Vec(myvocab,testEntry)#首先将文档转换为词词向量***
    class2=classifyNB(testvec, p0v,p1v,pab)
    print(testEntry,'classified as: ',class2)
#********主函数5*********
##文档词袋模型（词汇出现在文档中的次数不止一次）
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
#--------------------(示例二)使用朴素贝叶斯过滤垃圾邮件-----------------
def textParse(bigString):#文本解析(主要是分割字符串,并剔除部分异常词汇)
    import re
    wordlist=re.split(r'[^\w]',bigString)#将文本分隔为单个词汇,其中分隔符是除单词、数字外的任意字符串,[^ ]表示非字符集,[^\w]等价与\W
    return [word.lower() for word in wordlist if len(word)>2]#剔除单词长度小于2的,大写转小写
def spamTest():#邮件分类
    classlist=[];textlist=[]
    for i in range(1,26):
        spam=textParse(open('email/spam/%d.txt' % i).read())##'email/spam/%d.txt' %i 除了format的另外一种格式替换
        textlist.append(spam)
        classlist.append(1)
        ham=textParse(open('email/ham/%d.txt' % i).read())
        textlist.append(ham)
        classlist.append(0)
    vocablist=createVocabList(textlist)
    trainset=list(range(50));testset=[]
    for i in range(10):#重复抽取10次
        test=(int(random.uniform(0,len(trainset))))#随机产生0-50之间的数
        testset.append(test)
        del(trainset[test])#删除训练集中相应的数
    trainmat=[];trainclass=[]
    for index in trainset:
        trainmat.append(bagOfWords2VecMN(vocablist, textlist[index]))
        trainclass.append(classlist[index])
    p0v,p1v,pab=trainNB0(trainmat,trainclass)
    error=0
    for index in testset:
        testvector=bagOfWords2VecMN(vocablist, textlist[index])##测试集也需转换成词向量才能预测
        cla=classifyNB(testvector, p0v,p1v,pab)
        if cla != classlist[index]:
            error += 1
            print('class error is :',textlist[index])
    print('the error rate is: ',float(error)/len(testset))
#tips:随机选择测试集得出的错误率不稳定,应重复多次上述过程,然后对错误率求平均                        
#--------------------(示例三)使用朴素贝叶斯从个人广告中获取区域倾向----------------
import feedparser ##RSS阅读器
from numpy import *
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
##去除高频词(类似于移除停顿词)
def calcMostFreq(vocabList,fullText):
    import operator
    countlist={}
    for text in fullText:
        countlist[text]=fullText.count(text)#对所有单词计数
    sort=sorted(countlist.items(),key=operator.itemgetter(1),reverse=True)
    return sort[:30]
def localWords(feed1,feed0):##训练模型
    import feedparser
    minlen=min(len(feed1['entries']),len(feed0['entries']))
    textlist=[];fulltext=[];classlist=[]
    for i in range(minlen):
        wordlist=textParse(feed1['entries'][i]['summary'])
        textlist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(1)
        wordlist=textParse(feed0['entries'][i]['summary'])
        textlist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(0)
    vocablist=createVocabList(textlist)
    sort_30=calcMostFreq(vocablist,fullText)
    for i in sort_30:
        if i[0] in vocablist:
            vocablist.remove(i[0])#若该单词在前30,则剔除
    trainset=list(range(2*minlen));testset=[]
    for i in range(20):#重复抽取20次
        test=(int(random.uniform(0,len(trainset))))
        testset.append(test)
        del(trainset[test])#删除训练集中相应的数
    trainmat=[];trainclass=[]
    for index in trainset:
        trainmat.append(bagOfWords2VecMN(vocablist, textlist[index]))
        trainclass.append(classlist[index])
    p0v,p1v,pab=trainNB0(trainmat,trainclass)
    error=0
    for index in testset:
        testvector=bagOfWords2VecMN(vocablist, textlist[index])##测试集也需转换成词向量才能预测
        cla=classifyNB(testvector, p0v,p1v,pab)
        if cla != classlist[index]:
            error += 1
    print('the error rate is: ',float(error)/len(testset))
    return vocablist,p0v,p1v
##显示地域相关的用词
def getTopWords(ny,sf):
    topny=[];topsf=[]
    vocablist,p0v,p1v=localWords(ny,sf)
    for i in range(len(vocablist)):
        if p1v[i]>-6.0:#若概率大于-6.0？？？
            topny.append((vocablist[i],p1v[i]))
        if p0v[i]>-6.0:
            topsf.append((vocablist[i],p0v[i]))
    sortny=sorted(topny,key=operator.itemgetter(1),reverse=True)
    print('----ny-----ny------ny----')
    for i in sortny:
        print(i[0])
    sortsf=sorted(topsf,key=operator.itemgetter(1),reverse=True)
    print('----sf-----sf------sf----')
    for i in sortsf:
        print(i[0])
            
        
    
 
 
 
 
