#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/6 19:32
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : catering_sale.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #导入图像库
# from __future__ import print_function
from scipy.interpolate import lagrange

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


def correlation_analyze():
  '''
  计算菜品之间的相关系数
  :return:
  '''
  data = getDataFromExcel('./data/catering_sale_all.xls', u'日期')
  print(data.corr())  # 相关系数矩阵，即给出了任意两款菜式之间的相关系数
  print(data.corr()[u'百合酱蒸凤爪'])  # 只显示“百合酱蒸凤爪”与其他菜式的相关系数
  print(data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺']))  # 计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数


def getDataFromExcel(catering_sale_url,index_col):
  '''
  读取excel数据
  :param catering_sale_url: excel文件路径
  :param index_col: 索引列
  :return: excel数据
  '''
  # catering_sale_url = './data/catering_sale.xls'  # excel文件路径_餐饮数据
  data = pd.read_excel(catering_sale_url, index_col=index_col)  # 读取数据，指定“日期”列为索引列
  return data



def lagrange_interp():
  '''
  数据清洗
  用拉格朗日法进行插补
  :return:
  '''
  outputfile = './tmp/sales.xls' # 输出数据路径
  data = getDataFromExcel('./data/catering_sale.xls',None)
  data[u'销量'][(data[u'销量']<400) | (data[u'销量'] > 5000)] = None # 过滤异常值，将其变为空值

  for i in data.columns:
    for j in range(len(data)):
      if(data[i].isnull())[j]:
        data[i][j] = ployinterp_column(data[i],j)

  data.to_excel(outputfile)



def ployinterp_column(s,n,k=5):
  '''
  自定义列向量插值函数
  :param s: s为列向量
  :param n: n为被插值的位置
  :param k: k为取前后的数据个数，默认为5
  :return: 插值结果
  '''
  y = s[list(range(n-k,n)) + list(range(n+1, n+1+k))]
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) # 插值并返回插值结果



def data_normalization():
  '''
  数据规范化
  :return:
  '''
  datafile = './data/normalization_data.xls'
  data = pd.read_excel(datafile,header=None) # 读取数据

  print(data) # 原始数据
  print((data - data.min())/(data.max() - data.min())) # 最小-最大规范化
  print((data - data.mean())/data.std()) # 零-均值规范化
  print(data/10**np.ceil(np.log10(data.abs().max()))) # 小数定标规范化


def data_discretization():
  '''
  数据离散化
  :return:
  '''
  datafile = './data/discretization_data.xls'  # 参数初始化
  data = pd.read_excel(datafile)  # 读取数据
  data = data[u'肝气郁结证型系数'].copy()
  k = 4

  d1 = pd.cut(data, k, labels=range(k))  # 等宽离散化，各个类比依次命名为0,1,2,3

  # 等频率离散化
  w = [1.0 * i / k for i in range(k + 1)]
  w = data.describe(percentiles=w)[4:4 + k + 1]  # 使用describe函数自动计算分位数
  print(w)
  w[0] = w[0] * (1 - 1e-10)
  d2 = pd.cut(data, w, labels=range(k))

  from sklearn.cluster import KMeans  # 引入KMeans
  kmodel = KMeans(n_clusters=k, n_jobs=4)  # 建立模型，n_jobs是并行数，一般等于CPU数较好
  kmodel.fit(data.reshape((len(data), 1)))  # 训练模型
  c = pd.DataFrame(kmodel.cluster_centers_).sort(0)  # 输出聚类中心，并且排序（默认是随机序的）
  w = pd.rolling_mean(c, 2).iloc[1:]  # 相邻两项求中点，作为边界点
  w = [0] + list(w[0]) + [data.max()]  # 把首末边界点加上
  d3 = pd.cut(data, w, labels=range(k))

  cluster_plot(d1, k, data).show()
  cluster_plot(d2, k, data).show()
  cluster_plot(d3, k, data).show()


def cluster_plot(d, k, data):
  '''
  自定义作图函数来显示聚类结果
  :param d:
  :param k:
  :return:
  '''
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

  plt.figure(figsize=(8, 3))
  for j in range(0, k):
    plt.plot(data[d == j], [j for i in d[d == j]], 'o')

  plt.ylim(-0.5, k - 0.5)
  return plt



if __name__ == "__main__":
  # show_boxplot()
  # statistics_analyze()
  # catering_dish_profit()
  # correlation_analyze()
  # lagrange_interp()
  # data_normalization()
  data_discretization()
