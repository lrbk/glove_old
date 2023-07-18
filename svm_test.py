import pickle
import pandas as pd
import os
import tkinter as tk



hand='left'
num='withnum'
data_path='processed_data/test_data.csv'#输入路径

def main():
    global hand
    global num
    if hand=='left':
        len_data=15
    else:
        len_data=20

    # # lda_path='pretrained_model/lda_right_withnum.model'#lda模型
    # # model_path='pretrained_model/right_withnum.model'#model模型
    # # svm_path='pretrained_model/svm_right_withnum.model'
    # # lda_path='pretrained_model/lda_right_nonum.model'
    # # model_path='pretrained_model/right_nonum.model'
    # # svm_path='pretrained_model/svm_right_nonum.model'
    # # lda_path='pretrained_model/lda_left_nonum.model'
    # # model_path='pretrained_model/left_nonum.model'
    # # svm_path='pretrained_model/svm_left_nonum.model'
    # lda_path='pretrained_model/lda_left_withnum.model'
    # model_path='pretrained_model/left_withnum.model'
    # svm_path='pretrained_model/svm_left_withnum.model'

    # c_err=[('o','9'),('0','p'),('m','n'),('y','u','h'),('y','u'),('h','j'),('y','6'),('8','i')]  #right 
    # c_err=[('m','n'),('y','u','h'),('y','u'),('h','j')]  #right
    # c_err=[]  #左手带数字时，加上svm后准确率变低了
    # c_err=[('r','t'),('t','g')]

    lda_path = os.path.join('pretrained_model', f"lda_{hand}_{num}.model")
    model_path = os.path.join('pretrained_model', f"model_{hand}_{num}.model")
    svm_path = os.path.join('pretrained_model', f"svm_{hand}_{num}.model")
    if hand=='right':
        if num=='nonum':
            c_err=[('m','n'),('y','u','h'),('y','u'),('h','j')] 
        else:
            c_err=[('o','9'),('0','p'),('m','n'),('y','u','h'),('y','u'),('h','j'),('y','6'),('8','i')]
    else:
        if num=='nonum':
            c_err=[('r','t'),('t','g')]
        else:
            c_err=[] 

    # 加载模型
    with open(lda_path, 'rb') as f:
        lda_loaded = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(svm_path, 'rb') as f:
        svm_model = pickle.load(f)

    data = pd.read_csv(data_path,header=None)#对应的80维的数据
    X=[]
    for i in range(0,1):
        x=[]
        for j in range(0,8):#8行数据
            height=data.iloc[len_data*i:len_data*i+len_data,j].to_list()
            x.extend(height)
        X.append(x)

    xtest=lda_loaded.transform(X)
    output=model.predict(xtest)

    for k in range(len(c_err)):
        if output in c_err[k]:
            output=(svm_model[c_err[k]].predict(X))[0]
    # print(output)
    # 创建窗口
    window = tk.Tk()
    # 创建一个标签，用于显示字符
    window.title('键位')
    width = 380
    heigh = 150
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    window.geometry('%dx%d+%d+%d'%(width, heigh, (screenwidth-width)/2, (screenheight-heigh)/2))
    label = tk.Label(window, text=output,anchor="center",font="Helvetic 100 bold")
    label.pack()

    # 运行窗口的主循环
    window.mainloop()

if __name__ == '__main__':
    main()

