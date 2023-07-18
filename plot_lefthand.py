#from numba import jit
#from tkinter import *
#from tkinter import ttk
#from memory_profiler import profile

#import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports
import time
import datetime
import numpy as np
#import os
from time import sleep
# 导入CSV安装包
import csv
import codecs
#import readchar as rd
import random
import math

#定义串口接收  04 FF 1B 1B 05 00 00 00 15 17
#04 FF 1B 1B 05 00 00 00 15 17 00 01 C4 06 A0 0F A0 0F A0 0F 00 00 A0 0F A0 0F A0 0F A0 0F 

nn=50  #记录的噪声长度
lefthand=1

def recv(serial):
        while True:
                data =serial.read()#读一个字节数
                if data ==b'\x04':#如果读到第1个是04，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\xff':#如果读到第2个是0，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x1b':#如果读到第3个是0，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x1b':#如果读到第4个是0，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x05':#如果读到第5个是05，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第6个是0，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第7个是0，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第8个是0，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x15':#如果读到第9个是15，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x17':#如果读到第10个是17，继续读
                        data =serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第11个是00，继续读
                        data =serial.read(19)#读剩下所有字节数
                else:
                        break        
         
                break  
        return data


def receive_n_data(n): #接收n个data，并存入s_data返回(n*26)
    data1_13=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]                   
    s_data=[]                         
    while len(s_data)<n:
            d=list(range(0,26))
            data =recv(serial)
            if(len(data)>15 & data[0]==0):
                    for i in range (1,len(data)):
                            data1_13[i-1]=data[i]
            else:
                    if(len(data)>15 & data[0]==1):
                            for i in range (1,9):
                                    data1_13[17+i]=data[i]
                                    
                            for i in range (11,19):
                                    data1_13[43+i-10]=data[i]   
                    else:
                            if(len(data)>15 & data[0]==2):
                                    for i in range (1,19):
                                            data1_13[26+i-1]=data[i] 

                            for i in range(0,26):
                                    d[i]=data1_13[i*2+1]*256+data1_13[i*2]                                                   
                            s_data.append(d)
                            print(len(s_data),':',d)
    return s_data                              
  


if __name__ == '__main__':
#获取串口列表
        port_list = list(serial.tools.list_ports.comports())
        print(port_list)
        Open=0
        filename = 'data_new.csv' # 文件名
        
        if len(port_list)==0:
                print('无可用串口')
        else:
                for i in range(0,len(port_list)):
                        print(port_list[i])
                        if('COM9' in port_list[i]):
                                serial = serial.Serial('COM9', 115200, timeout=200)  #填入实际串口号
                                print("serial open success")
                                Open=1
        if lefthand==1:
                L=['1','2','3','4','5','q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
        else:
                L=['6','7','8','9','0','y','u','h','j','n','m','i','k','l','o','p','+']

        label=[]
        if (Open==1) :
                #
                print('下面开始测试传感器是否完好以及噪声：')         
                print('请将手套保持不动10秒，记录噪声')
                noise=receive_n_data(nn)
                avg=[]
                sigma=[]
                right=13
                #
                #记录噪声
                print('噪声已读取完毕，按回车后继续')
                with open('noise2.csv','a+',newline="") as f7:
                        writer=csv.writer(f7)
                        for line in noise:
                                writer.writerow(line)
                print("  ",end=" \n")
                input()
                #
                for i in range(8):
                        dn=[line[i+right] for line in noise]
                        a1=sum(dn)/nn
                        avg.append(a1)
                        s=0
                        for k in range(nn):
                                s+=(dn[k]-a1)*(dn[k]-a1)
                        s=s/nn
                        sigma.append(math.sqrt(s))    #总体标准差
                print('均值：',avg)
                print('标准差：',sigma)
                #
                print('按回车后继续检查传感器')
                input()
                #

                while(1):
                    print('请输入检查第几个传感器：（1，2，...,8）,输入9退出检查')
                    x=int(input())
                    if x not in range(1,10):
                        print('不存在该传感器编号')
                    else:
                        if x==9:
                                print('已退出检查')
                                break
                        else:
                            print('正在检查第',x,'个传感器：')
                            print('按下0后移动传感器')
                            y=input()
                            if y=='0':
                                print('再次按下0，随后移动该传感器')
                                t_data=receive_n_data(50)
                                avg_t=[]
                                for i in range(8):
                                    n11=[line[i+right] for line in t_data]
                                    an1=sum(n11)/len(n11)
                                    avg_t.append(an1)
                                    m=0
                                for i in range(8):
                                        q=abs(avg_t[i]-avg[i])
                                        if i!=x-1:                                              
                                                if q>10:
                                                        print('第',i+1,'个传感器动了，检查不通过')
                                                        m=1
                                                        break
                                        else:
                                                if q<10:
                                                       print('第',x,'个传感器检查不通过')
                                                       m=1
                                                       break
                                if m==0:
                                       print('第',x,'个传感器检查完成')
                                else:
                                       print('已退出检查')
                                       break
                                        
                                
                            else:
                                 print('错误按键，请重新输入传感器编号')  
                                 continue


                #

                ii=1
                while(1):                   
                    a=random.choice(L)
                    print('下面开始正式测试，请进行手势：',a)
                    print('第',ii,'次，','按下 - 后开始记录')#按下=后终止程序,-开始记录数据，[确认保存数据，]取消保存数据
                    x=input()
                    #
                    print('再次按下回车后继续')
                    input()
                    #
                    if x=='=':
                            print('程序已退出')
                            print(label)
                            input()
                            break
                    s_data=[]           
                    data1_13=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    
                    if x=='-':
                                s_data=receive_n_data(100) #100行一个手势

                                print('是否确认记录按键',a,'    是：[    否：]')                                
                                b=input()
                                if b=='[':
                                        print('正在记录第',ii,'次手势')
                                        ii+=1
                                        label.append(a)
                                        with open(filename,'a+',newline="") as f1:
                                                writer=csv.writer(f1)
                                                for line in s_data:
                                                        writer.writerow(line)
                                        print("  ",end=" \n")

                                        # #幅度归一化
                                        # dealed=[]                                        
                                        # for jj in range(len(s_data)):
                                        #         line=[]
                                        #         for kk in range(8):
                                        #                 line.append(s_data[jj][kk+right]-avg[kk])
                                        #         dealed.append(line)   

                                        # with open("dealed_data.csv",'a+',newline="") as f3:
                                        #         writer=csv.writer(f3)
                                        #         for line in dealed:
                                        #                 writer.writerow(line)
                                        # print("  ",end=" \n")                                                                            
                                        # #                         
                                else:
                                        if b==']':
                                                continue
                                        else:
                                                if b=='=':
                                                        print('程序已退出')
                                                        print(label)
                                                        input()
                                                        break
                                                
                with open("label_new.csv", "a+", newline="") as f2:
                        for j in range(len(label)):
                                csv_writer = csv.writer(f2)   
                                csv_writer.writerow(label[j])
                        csv_writer.writerow("\n")
                        # f2.close()
                print(" ",end=' \n')
                print('已记录label')
                print(label)

        else:
                print("serial open failed")   
        input()                         
        
