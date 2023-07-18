from collections import Counter
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score,cross_validate,train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import minmax_scale,StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

data_path1='./datas/left_nonum_data.csv'
label_path1='./datas/left_nonum_label.csv'
# data_path2='./data_new/datas/c1_data27.csv'
# label_path2='./data_new/datas/s1_label27.csv'
data_path2='./error_data/T.csv'
data1 = pd.read_csv(data_path1,header=None)#对应的80维的数据
data2 = pd.read_csv(data_path2,header=None)
y1 = pd.read_csv(label_path1,header=None)#对应的label,len应该和X相同
num1=len(y1)
X1=[]
for i in range(0,num1):
    x=[]
    for j in range(0,8):#8行数据
        height=data1.iloc[i:i+1,j].to_list()
        x.extend(height)
    X1.append(x)
y1=y1.values.ravel()
df=pd.DataFrame(X1)
df['Label'] = y1
group1 = df[df['Label'] == 't'].drop(columns='Label')
mean1=np.mean(group1,axis=0)
mean2=np.mean(data2,axis=0)
print(mean1)
print(mean2)
print(np.corrcoef(mean1,mean2))