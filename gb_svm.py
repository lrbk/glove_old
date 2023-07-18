# # ##由于样本自带标签，初步设想是用LDA方法进行降维
from collections import Counter
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import minmax_scale,StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt 
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pickle
import os


hand='right'
num='nonum'

data_path=os.path.join('datas',f"{hand}_{num}_data.csv")
label_path=os.path.join('datas',f"{hand}_{num}_label.csv")
lda_path = os.path.join('pretrained_model', f"lda_{hand}_{num}.model")
model_path = os.path.join('pretrained_model', f"model_{hand}_{num}.model")
svm_path = os.path.join('pretrained_model', f"svm_{hand}_{num}.model")

#设置c_err和L
if hand=='right':
    if num=='nonum':
        c_err=[('m','n'),('y','u','h'),('y','u'),('h','j')] 
        L=['y','u','h','j','n','m','i','k','l','o','p','+']
    else:
        c_err=[('o','9'),('0','p'),('m','n'),('y','u','h'),('y','u'),('h','j'),('y','6'),('8','i')]
        L=['6','7','8','9','0','y','u','h','j','n','m','i','k','l','o','p','+']
else:
    if num=='nonum':
        c_err=[('r','t'),('t','g')]
        L=['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
    else:
        c_err=[] 
        L=['1','2','3','4','5','q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']

#所降低到的维度的范围
if hand=='left':  #不带数字  #带数字：right:14,  left:17
    min_dim=14
    len_data=15  #波的长度
else:
    min_dim=11
    len_data=20
max_dim=min_dim+1
err=0.01  #加入的噪声方差
n_datas=15  #数据扩增的增加倍数
nsplits=5  #交叉验证的折数
nerouns=(100,) #隐藏层神经元个数，参数hidden_layer_sizes
lr=0.1  #初始学习率
mm='gaussgb'  #gaussgb or mlp


# define dataset
data = pd.read_csv(data_path,header=None)#对应的80维的数据
y = pd.read_csv(label_path,header=None)#对应的label,len应该和X相同
# test_data = pd.read_csv(test_data_path,header=None)

num=len(y)
# y0=[]
y=y.values.ravel()

one_hot = LabelBinarizer()  
# L0=one_hot.fit_transform(L) 
L0=one_hot.fit(L)

print('datapath:',data_path)
X=[]
for i in range(0,num):
    x=[]
    for j in range(0,8):#8行数据
        height=data.iloc[len_data*i:len_data*i+len_data,j].to_list()
        # height_scale=minmax_scale(height)
        # x.extend(height_scale)
        x.extend(height)
    X.append(x)

#训练集数据扩增

# models = get_models()
times=0
for dim in range(min_dim,max_dim):
    print('降维后维度',dim)
    kf = KFold(n_splits=nsplits)
    train_score=[]
    test_score=[]
    train_svm_score=[]
    test_svm_score=[]
    tr_or=[]
    te_or=[]
    gt_labels=[]
    pred_labels=[]
    # if True:
    for train,test in kf.split(X):
        print('times:',times)
        times+=1
        Xtrain0=[]
        xtest0=[]
        Ytrain=[]
        ytest=[]
        Ytrain0=[]
        ytest0=[]
        #print (k,(train,test))
        for val in train:
            Xtrain0.append(X[val])
            Ytrain.append(y[val])
            # Ytrain0.append(y0[val])
        for val in test:
            xtest0.append(X[val])
            ytest.append(y[val])   

        #先降维，后数据扩增
        lda=LDA(n_components=dim)
        lda.fit(Xtrain0,Ytrain)
        Xtrain=lda.transform(Xtrain0)
        xtest=lda.transform(xtest0)
        
        # 保存LDA模型
        with open(lda_path, 'wb') as f:
            pickle.dump(lda, f)

        xtr_or=Xtrain
        ytr_or=Ytrain

        n_or=len(ytr_or)
        n=len(Ytrain)
        data=dict()
        for label in L:
            label_set=[]
            for ii in range(n):
                if Ytrain[ii]==label:
                    label_set.append(Xtrain[ii])
            data[label]=label_set

        Xtrain=[]
        Ytrain=[]
        for i in range(len(L)):
            label=L[i]
            mean=np.mean(data[label],axis=0)
            cov=np.cov(np.transpose(data[label]))
            n_or=len(data[label])
            #print('扩增了%d个数据' %(n_or*n_datas))
            new_train = np.random.multivariate_normal(mean, cov, size=(n_datas*n_or))
            data[label]=np.vstack((data[label],new_train))
            n=len(data[label])
            if label==L[0]:
                Xtrain=data[label]
            else:
                Xtrain=np.vstack((Xtrain,data[label]))
            for i in range(n):
                Ytrain.append(label)
        # print(Xtrain.shape)

        #打乱训练集的数据顺序
        ii=[i for i in range(len(Xtrain))]
        random.shuffle(ii)
        xx=[]
        yy=[]
        for i in ii:
            xx.append(Xtrain[i,:])
            yy.append(Ytrain[i])
        xx=np.array(xx,dtype=float)

        #模型为bp神经网络  
        #数据集Xtrain,xtest，xtr_or等已被lda降维过
        # #model = MLPRegressor(hidden_layer_sizes=(1,), random_state=10,learning_rate_init=0.1)  # BP神经网络回归模型
        if mm=='mlp':
            model = MLPClassifier(hidden_layer_sizes=nerouns,learning_rate_init=lr)  # BP神经网络回归模型
            model1= MLPClassifier(hidden_layer_sizes=nerouns,learning_rate_init=lr)
        if mm=='gaussgb':
            model=GaussianNB()
            model1=GaussianNB()
        #one-hot编码标签
        
        if mm=='mlp':
            #以下为mlp
            y0=one_hot.transform(yy)
            y0test=one_hot.transform(ytest)
            y0tr_or=one_hot.transform(ytr_or)
        if mm=='gaussgb':
            #以下为gaussgb
            y0=yy
            y0test=ytest
            y0tr_or=ytr_or
        #

        model.fit(xx,y0)  # 训练模型(mlp)
        # model.fit(xx,yy)  #(Gaussgb)
        model1.fit(xtr_or,y0tr_or)

        pre = model.predict(xtest)  # 模型预测 
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        # gt_labels.extend(y0test)
        # pred_labels.extend(pre)
        # print(np.abs(y0test-pre).mean())

        # if mm=='gaussgb':
        #     #model=LDA(n_components=dim).fit_transform(xx)
        #     model=GaussianNB().fit(xx,yy)

        score=model.score(xx,y0)  #mlp
        # score=model.score(xx,yy)  #gaussgb
        train_score.append(score)
        # print('train score:',score)
        score=model.score(xtest, y0test)  #mlp
        # score=model.score(xtest, ytest)  #gaussgb
        test_score.append(score)

        # if mm=='mlp':
        #     model=MLPClassifier(hidden_layer_sizes=(100,),learning_rate_init=0.1).fit(xtr_or,y0tr_or)
        # if mm=='gaussgb':
        #     model=GaussianNB()
        score=model1.score(xtr_or,y0tr_or)  #mlp
        # score=model.score(xtr_or,ytr_or)  #gaussgb
        #混淆矩阵
        pre_train=model1.predict(xtr_or)
        pre_test=pre
        gt_labels.extend(y0test)  #修改y0tr_or（数据扩增前，训练集）或y0test（数据扩增前，测试集）
        # pred_labels.extend(pre_svm_test)   #修改pre_train（数据扩增前，训练集）或pre_test（数据扩增前，训练集）  或pre_svm_test  (svm)
        tr_or.append(score)
        score=model1.score(xtest, y0test)  #mlp
        # score=model.score(xtest, ytest)  #gaussgb
        te_or.append(score)    # print('train score:',train_score)
        # print('未数据扩增，未svm，测试集准确率：',score)


        #svm
        id_svm={}
        for j in range(len(y0tr_or)):
            for i in range(len(c_err)):
                if y0tr_or[j] in c_err[i]:
                    ids=id_svm.get(c_err[i],[])
                    ids.append(j)
                    # if y[j]=='o':
                    #     print('o:,y[j],y0r_or[j]:',j,y[j])
                    id_svm[c_err[i]]=ids  
        # print('y0tr_or[-20:-1]:',y0tr_or[-20:-1])           
        # print('id_svm[0:20]:',id_svm[c_err[0]][0:20])
        xtr_svm=[]
        ytr_svm=[]
        for k in range(len(c_err)):
            for key,value in id_svm.items():
                if key==c_err[k]:
                    data_svm=[]
                    lb_svm=[]
                    for v in value:
                        data_svm.append(Xtrain0[v])  #Xtrain0：未降维，未数据扩增
                        lb_svm.append(y0tr_or[v])
                    xtr_svm.append(data_svm)
                    ytr_svm.append(lb_svm)
                    # print('lenth of lbs:',len(lb_svm))
        # print('ytr_svm:',ytr_svm[0][0:10])
        svm_model={}
        for k in range(len(c_err)):
            if len(c_err[k])==2:
                # steps = [('pca', PCA()), ('svm', LinearSVC())]       
                # svm_model[c_err[k]]=Pipeline(steps=steps)
                svm_model[c_err[k]]=LinearSVC()
            else:
                # steps = [('lda', LDA(n_components=len(c_err[k])-1)), ('m', GaussianNB())]       
                # steps = [('lda', LDA(n_components=len(c_err[k])-1)), ('k', tree.DecisionTreeClassifier())]       
                # svm_model[c_err[k]]=Pipeline(steps=steps)
                # svm_model[c_err[k]]=KNeighborsClassifier()
                svm_model[c_err[k]]=LinearSVC()
                # svm_model[c_err[k]]=tree.DecisionTreeClassifier()
            svm_model[c_err[k]].fit(xtr_svm[k],ytr_svm[k])
        # print('svm_model:',svm_model)
        with open(svm_path, 'wb') as f:
            pickle.dump(svm_model, f)
        #predict svm
        pre_svm_test=pre_test
        pre_svm_train=pre_train
        for k in range(len(c_err)):
            for i in range(len(pre_test)):
                if pre_test[i] in c_err[k]:
                    # idpre.append[i]  #测试集xtest0(未降维)，y0test  （xtest已降维）
                    # print('svm predict:',svm_model[c_err[k]].predict([xtest0[i]]))
                    pre_svm_test[i]=(svm_model[c_err[k]].predict([xtest0[i]]))[0]
        for k in range(len(c_err)):
            for i in range(len(pre_train)):
                if pre_train[i] in c_err[k]:
                     #训练集Xtrain0(未降维,未数据扩增)，y0tr_or  （xtr_or已降维）
                    # print('svm train predict:',svm_model[c_err[k]].predict([Xtrain0[i]]))
                    pre_svm_train[i]=(svm_model[c_err[k]].predict([Xtrain0[i]]))[0]
        
        pred_labels.extend(pre_svm_test)
        #计算准确率
        count=0
        for i in range(len(y0tr_or)):
            if pre_svm_train[i]==y0tr_or[i]:
                count+=1
        train_svm_score.append(count/len(y0tr_or))
        print('train_svm_score:',count/len(y0tr_or))
        count=0
        for i in range(len(y0test)):
            if pre_svm_test[i]==y0test[i]:
                count+=1
        test_svm_score.append(count/len(y0test))
        print('test_svm_score:',count/len(y0test))








    # print('test score:',test_score)
        # print(np.abs(y0test-pre).mean())
    # print('hidden_layer_size=',nerouns)
    print('数据扩增后：(扩增倍数%d)' %n_datas)
    print('mean train score:',sum(train_score)/nsplits)
    print('mean test score:',sum(test_score)/nsplits)
    print('数据扩增前：' )
    # print('train score:',tr_or)
    # print('test score:',te_or)
    print('mean train score:',sum(tr_or)/nsplits)
    print('mean test score:',sum(te_or)/nsplits)  

    print('未数据扩增，经过svm后：')
    print('mean train score:',sum(train_svm_score)/nsplits)
    print('mean test score:',sum(test_svm_score)/nsplits)     
    # id_svm={}
    # for j in range(len(y)):
    #     for i in range(len(c_err)):
    #         if y[j] in c_err[i]:
    #             ids=id_svm.get(c_err[i],[])
    #             ids.append(j)
    #             id_svm[c_err[i]]=ids             
    # print('id_svm:',id_svm)




    # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix

   
    # print(len(gt_labels))
    # confusion_mat = confusion_matrix(gt_labels, pred_labels,labels=L)
    # c_percent=confusion_mat/ confusion_mat.astype(np.float64).sum(axis=1)*100
    # c_percent=c_percent.astype(np.int_)
    # # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=L)
    # disp = ConfusionMatrixDisplay(confusion_matrix=c_percent, display_labels=L)
    # disp.plot(
    #     include_values=True,            # 混淆矩阵每个单元格上显示具体数值
    #     cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
    #     ax=None,                        # 同上
    #     xticks_rotation="horizontal",   # 同上
    #     values_format="d"               # 显示的数值格式
    # )
    # plt.savefig('./new_train_set2/right_svm_nonum_test6.png')
    # break