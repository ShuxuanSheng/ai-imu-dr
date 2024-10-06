import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件，没有列名时
data1 = pd.read_csv('output_filter.csv', header=None)  # header=None 指定没有列名
data2 = pd.read_csv('output_fixed.csv', header=None)

x1, y1 = data1.iloc[:, 1], data1.iloc[:, 2]  # 第一列和第二列
x2, y2 = data2.iloc[:, 1], data2.iloc[:, 2]  # 第一列和第二列

# 绘图
plt.figure()
plt.scatter(x1, y1, label='output_filter', color='blue')
plt.scatter(x2, y2, label='output_fixed', color='red')

# 添加标签和图例
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('XY坐标图')
plt.legend()
plt.grid()

# 显示图形
plt.show()
