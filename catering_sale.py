#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/6 19:32
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : catering_sale.py
# @Software: PyCharm


import pandas as pd
import matplotlib.pyplot as plt #导入图像库
# from __future__ import print_function


def show_boxplot():
  '''
  利用箱线图检测餐饮销售数据异常值
  :return:
  '''

  data = getDataFromExcel('./data/catering_sale.xls',u'日期')
  print(data.describe())

  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

  plt.figure()  # 建立图像
  p = data.boxplot(return_type='dict')  # 画箱线图，直接使用DataFrame的方法
  print(p)

  # 'flies'即为异常值的标签
  x = p['fliers'][0].get_xdata()
  y = p['fliers'][0].get_ydata()
  y.sort()  # 从小到大排序，该方法直接改变原对象

  print(x)

  # 用annotate添加注释
  # 其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
  # 以下参数都是经过调试的，需要具体问题具体调试。
  for i in range(len(x)):
    if i > 0:
      plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / (y[i] - y[i - 1]), y[i]))
    else:
      plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.08, y[i]))

  plt.show()  # 展示箱线图


def statistics_analyze():
  '''
  餐饮销量数据统计量分析
  :return:
  '''
  data = getDataFromExcel('./data/catering_sale.xls',u'日期')
  print(data.describe())

  data = data[(data[u'销量'] > 400) & (data[u'销量'] < 5000)]  # 过滤异常数据
  statistics = data.describe()  # 保存基本统计量

  statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']  # 极差
  statistics.loc['var'] = statistics.loc['std'] / statistics.loc['mean']  # 变异系数
  statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']  # 四分位数间距

  print(statistics)


def catering_dish_profit():
  '''
  菜品盈利数据 帕累托图
  :return:
  '''

  data = getDataFromExcel('./data/catering_dish_profit.xls',u'菜品名')
  # 初始化参数
  data = data[u'盈利'].copy()
  # data.sort(ascending=False)
  data.sort_values(inplace=True,ascending=False)

  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

  plt.figure()
  data.plot(kind='bar')
  plt.ylabel(u'盈利（元）')
  p = 1.0 * data.cumsum() / data.sum()
  p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
  plt.annotate(format(p[6], '.4%'), xy=(6, p[6]), xytext=(6 * 0.9, p[6] * 0.9),
               arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))  # 添加注释，即85%处的标记。这里包括了指定箭头样式。
  plt.ylabel(u'盈利（比例）')
  plt.show()


def getDataFromExcel(catering_sale_url,index_col):
  # catering_sale_url = './data/catering_sale.xls'  # excel文件路径_餐饮数据
  data = pd.read_excel(catering_sale_url, index_col=index_col)  # 读取数据，指定“日期”列为索引列
  return data


if __name__ == "__main__":
  # show_boxplot()
  # statistics_analyze()
  catering_dish_profit()


