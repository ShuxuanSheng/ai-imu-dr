import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件，没有列名时
data1 = pd.read_csv('p_gt.csv', header=None)  #p_gt
data2 = pd.read_csv('output_fixed.csv', header=None) #p_fixed
data3 = pd.read_csv('output_filter.csv', header=None) #p_filter
data4 = pd.read_csv('output_wheel_fixed.csv', header=None) #p_wheel


x1, y1, z1 = data1.iloc[:, 0], data1.iloc[:, 1], data1.iloc[:, 2]
x2, y2, z2 = data2.iloc[:, 1], data2.iloc[:, 2], data2.iloc[:, 3]
x3, y3, z3 = data3.iloc[:, 1], data3.iloc[:, 2], data3.iloc[:, 3]
x4, y4, z4 = data4.iloc[:, 1], data4.iloc[:, 2], data4.iloc[:, 3]


# 绘图
fig1, axs1 = plt.subplots()
plt.plot(x1,y1, label='p_gt', color='blue')
plt.plot(x2,y2, label='output_fixed', color='red')
plt.plot(x3,y3, label='output_filter')
plt.plot(x4,y4, label='output_wheel_fixed')
axs1.text(x1.iloc[0], y1.iloc[0], 'Start', fontsize=12, ha='right')
axs1.text(x1.iloc[-1], y1.iloc[-1], 'End', fontsize=12, ha='right')
# 添加标签和图例
plt.xlabel('X')
plt.ylabel('Y')
plt.title('XY')
plt.legend()
plt.grid()

# 显示图形
plt.show()
