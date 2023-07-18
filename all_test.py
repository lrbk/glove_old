import os
import plot_lefthand
import last_plot
import deal_22
import cubic_regular11
import regular_whole
import svm_test
#以下两个变量是需要根据情况更改的
# hand表示左右手，有left和right两种值。
# num表示有无数字，有withnum和nonum两种取值
hand='left'
num='withnum'

#这部分代码只能在有采集设备的电脑上运行
if hand=='left':
    # os.system("python plot_lefthand.py")
    plot_lefthand.main()
    
elif hand=='right':
    os.system("python last_plot.py")
# 以上是输入端
##运行初始输入程序，将结果保存到 'data_new.csv'

#以下代码不用更改
deal_22.hand=hand
deal_22.main() 
#找到波的存在
cubic_regular11.hand=hand
cubic_regular11.main()
#归一化波
regular_whole.hand=hand
regular_whole.num=num
regular_whole.main()
# 整体归一化
svm_test.hand=hand
svm_test.num=num
svm_test.main()
#对于输入进行预测，返回最后的预测键位


