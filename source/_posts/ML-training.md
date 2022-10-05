---
title: Python数据分析2
tags: []
date: 2019-03-15 10:45:53
permalink:
categories: Data Analysis
description: 记录了Python, Numpy, Pandas, Hadoop, Spark等基本使用
image:
---
<p class="description">"Python数据分析1"与"Python数据分析2"分别是两门课的笔记，可能有相同之处</p>


<!-- more -->
# <font color=red>jupyter notebook技巧</font>
## 1. 安装jupyter notbook及其扩展
```Python
conda install jupyter notebook # 安装anaconda之后，下载jupyter notebook
conda install -c conda-forge jupyter_contrib_nbextensions # jupyter notebook扩展安装
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple autopep8 # 代码自动规范化包
```
[jupyter extension使用参考](https://zhuanlan.zhihu.com/p/52890101)
## 2. jupyter使用设置
```Python
# 设置输入输出情况
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'/'last_expr'
# matplotlib画图显示
%matplotlib inline
# 使用bash命令
!head -n 5 xx.txt
!ls -ltrh *
```
## 3. pyhton2 python3 kernel共存
```python
1. 安装anaconda3
2. conda create -n py27 python=2.7
3. source activate py27
4. conda install -n py27 ipykernel
5. python -m ipykernel install --user --name py27 --display-name "python2"
```
## 4. 本地访问远端服务器
```bash
# sever端
jupyter notebook --allow-root --port=8889 --no-browser
# local端
ssh -f -N -L localhost:8000:localhost:8889 xxx@xx.xx.xx.xx -p 6666
# 打开网页
http://localhost:8000
```


# <font color=red>Python包Pandas学习笔记</font>
__最近对pandas做了一个系统的学习，主要参考了这套[习题](https://github.com/guipsamora/pandas_exercises)，方便后续进行python数据分析__

## 1 package的载入
```Python
import numpy as np
import pandas as pd
```

## 2 数据读取(“/”暂作“或”使用)
```Python
# \s+是利用空白符进行分割,可使用正则表达式匹配
# header可以制定列明对应的行数，如果没有就是None
# index_col就是行名对应的列数，如果没有也是None
# parse_dates日期型列
pd.read_csv(path, sep='\s+'/'\t'/',', header=0/None, usecols=[...], 
	,parse_dates=[0]/['Date'], index_col=0/None, encoding='utf-8')
```

## 3 数据输出
```Python
# na_rep是填充的缺失值
# header可以给一个list用作数据输出的列名
# columns可以给定数字list用于选取输出特定的列
pd.to_csv(path, sep='', na_rep='NA', header=True/['xx','xx'], index=True, 
	columns=[1,2,3..], encoding='utf-8')
```

## 4 创建DataFrame,Series对象
```Python
# DataFrame和Series都接受list和numpy array数据，若要转换回array可以用df.values或S.values
# DataFrame还可以接受dictionary对象
df = pd.DataFrame(np.random.rand(10,5), columns=list('ABCDE')) #列名为ABCDE
S = pd.Series([1,2,3,4])
```

## 5 数据类型总结
```Python
df = pd.DataFrame({'string': list('abc'),
                   'int64': list(range(1, 4)),
                   'uint8': np.arange(3, 6).astype('u1'),
                   'float64': np.arange(4.0, 7.0),
                   'bool1': [True, False, True],
                   'bool2': [False, True, False],
                   'dates': pd.date_range('now', periods=3).values,
                   'category': pd.Series(list("ABC")).astype('category')})
S.astype(np.int64) # 数据类型转化
S.dtype == np.int64 # 数据类型判断
S.to_frame() # 将Series数据转换为DataFrame数据
pd.to_datetime(df.a) # 将df数据a列转换为datetime类型
pd.to_numeric(aDF.loc[:, 'xxx'])  #转换成数字
```

## 6 查看数据
```Python
df.head(10) # 查看数据前10行
df.tail(10) # 查看数据后10行
df.shape # 数据对应的行列数
df.info() # 数据索引，类型，内存信息
df.describe(include='all') # 对于数字型数据进行分位数统计,all代表对所有数据类型统计
S.value_counts(dropna=False) # 对Series里面的值进行个数统计，NA也会统计
df.apply(pd.Series.value_counts) #返回每个值在各列中的个数，没有则用NaN代替
df.index # 返回数据的索引
df.columns # 返回数据的列名
df.mean()/min()/median()/std()/count() # 分别是df的列均值,最小值,中位数,标准差,非空值
df.corr() # 列之间的相关系数
df.idxmax(0) # 每列最大数对应行的index的名
df.idxmax(1) # 每行最大数对应的列名
aDF.nsmallest(columns='age', n=20) # 取出年龄最小的20个数据
S.is_unique # 确定Series数据是否是unique的, 返回bool值
```

## 7 数据的截取
```Python
df.xxx / df['xxx'] # 返回数据的某一列，列名为xxx，类型为Series
df[['xx','yy']] # 选取多列数据，类型为DataFrame
df.iloc[3] # 选取第4行数据
df.iloc[[1,2,3],:] # 选取多行数据
df.loc[['x','y','z'],['a','b','c']] #根据行index名和列名进行数据截取
df.ix[..,..] # ix是用行名列名以及行数列数混合赋值的情况下数据的截取
df[(df.col>0.5) & (df.col<10)] # 筛选col大于0.5小于10的行返回
aDF.loc[(aDF['a']>10) & (aDF['b']<100), :] # 也可以给条件进行筛选,& | ~进行逻辑运算
df.values # 返回df对应的numpy array值
df.values[10][5] #求df数据11行6列的值
df[df['xxx'].notnull(), :] = 10 # 对空值赋值
S.str.startswith('G') # Seriers以G开头的字符, 返回bool值
S.isin(['a','b','c']) # 返回bool值如果S在list['a','b','c']中, 返回true
S != 'xxx' # Series中不为xxx的位置, 返回bool值
```

## 8 数据清洗
```Python
del df['a'] #删除数据df的a列
df.drop(['B','C'],axis=1,inplace=True) # 删除数据df的B,C列, 在原数据上进行修改
df.dropna(how = 'all', axis=0) # 对行进行操作,如果一行里面全都是na那么就删除,如果要一列里面全是na就删除,那么axis=1df.loc[:, 'newcol'] = 2000 # 如果没有newcol那么就新加一列
df.columns = ['a','b'] # 更改数据的列名
df.isnull().sum() # 统计各列中缺失值的个数
df.notnull().sum() # 统计各列中非缺失值的个数
df.drop_duplicates([...], 'first'/'last'/False) 
# 移除重复项, list可以是列名也可以是数字序列,first是保留重复中的第一个, last是保留重复中的最后一个, False一个不留, 只要重复都删除
df.dropna(axis=0/1, how='any'/'all', thresh=n)
# 移除数据中的缺失行,axis0是行删除,1是列删除
# any是只要有一个就删除,all是所有的都是才删除
# thresh目的是大于n个NA才删除
S.fillna(x) # 对确实值进行填充,填充值为x
aDF.loc[:,'xx'].fillna(aSeires) # 对数据进行填充,根据aSeires的index和aDF的index进行填充
S.astype(np.float) # 数据的类型转换
S.replace(1, 'one') # 将1替换为'one'
S.replace([1,2], ['one','two']) # 将1替换为one, 2替换为two
df.rename(index/columns={'old1':'new1','old2':'new2'}) # 修改行名列名，将old1改为new1，将old2改为new2
df.set_index('B') # 修改index，将B所在的列作为行索引
df.sort_index() # 将数据按照index进行排序
df.sort_values([col1, col2], ascending=[True,False]) # 根据col1和col2的值进行排序，col1是升序，col2是降序
S.argsort() # 返回Series对应的值的order，S[S.argsort()]返回的是对应的从小到大的Series数值
df.reset_index(drop=False/True, inplace=False/True) # 数据的index换成从0开始的, drop是说是否保留原来的index, 保留的话就多一列, inplace是说是否修改原来的df
data['Age'] = pd.cut(data['Age'], bins=6, labels=np.arange(6)) # 对数值型数据进行区间分割，分割成6个bin，label用0-5表示
np.tile(a, N).flatten() # 对数据a进行重复

```

## 9 数据分组
```Python
df.groupby(col).size() # 按照col对数据进行分组，并计算每一组的数量, 如果是count的话每列都会计算一次分组后的数量, 比较冗余
df.groupby([col1, col2]).mean() # 按照col1,col2进行分组，并计算各组的均值
df.groupby([col1, col2]).agg(['min', 'max', 'mean']) # 按照col1col2进行分组，计算各组之间的min, max, mean
df.groupby([col1, col2]).agg({'a':'min', 'b':'max', 'c':'mean'}) # 按照col1col2进行分组，计算各组之间的a列的min, b列的max, c列的mean
df.groupby(col1)[col2].mean() # 计算按照col1分组的col2对应的均值
df.pivot.table(index=col1, values=[col2, col3], aggfun='mean') # 以col1的值为index,以col2,col3值为列进行分组计算各元素平均值
df.apply(np.mean) # 计算每一列的平均值
df.apply(np.max, axis=1) #计算每一行的平均值
df.applymap(lambda x : x.upper()) # 对多列数据操作, 这里是对各列都大写
for name,data in df.groupby('col'):
	print name # col列分类后的值
	print data # col列名称等于name对应的数据行
df.groupby('name').apply(ownfunc) # 可以对不同组进行自定义函数操作
```

## 10 数据合并
```Python
df1.append([df2, df3])  # 也可以追加两个DF
pd.concat([df1.set_index('a'), df2.set_index('a')], sort=False, axis=1, join='inner') # 和上述利用merge在a字段上进行内连接的效果类似,因为concat是基于index进行连接的,merge可以不基于index,指定字段
pd.concat([df1, df2], axis=1) #列数相同, 行合在一起
pd.concat(frames, keys=['x', 'y', 'z'], axis=0) #行数相同, 列合在一起, 每个数据的来源分别标注xyz
pd.merge(df1, df2, on=['key1','key2'], how='outer'/'inner'/'left'/'right') # 合并df1和df2,根据df1的key1和df2的key2, 连接方式是外链接..
pd.merge(df1, df2, how='inner', left_index=True, right_on='id') # 对数据进行merge,左表以index作为连接关键字,右表用id作为关键字
np.vstack((a,b)) # 乱入一个numpy合并用法，行合并
# pd.join与merge用法类似, 只不过默认是left链接, merge是inner连接
```

<div style='display: none'>
### 2.1 div的用法
```Python
import numpy as np
import pandas as pd
users = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', 
        	sep='|', index_col='user_id')
total = users.groupby(['occupation']).gender.count() # 计算occupation的对应的人数
gender = users.groupby(['occupation','gender']).gender.count() # 计算各职业各性别的人数
(gender.div(total, level='occupation')*100).unstack() #计算各职业的各性别的百分比
```
</div>

## 11 时序数据操作
```Python
df.index = pd.date_range('2018/1/1', period=df.shape[0]) # 添加时间序列作为行名
pd.to_datetime(df.a, format='%Y') # 将df的a列转化成datetime类型的年
pd.to_datetime(1490195805, unit='s') # 对UNIX时间进行时间转换
df['a'].to_datetime().year/month/day # df的a列转换为datetime类型之后提取其中的year或者month或者day
(df['a'].to_datetime().max() - df['a'].to_datetime().min()).days # 计算a列中最早最晚时间差
df['Date'].dt.dayofweek # 获取每周第几天

import datetime as dt
dt.datetime(2015,1,1) # 通过datetime包定义数据时间
dt.datetime.today() # 返回今天对应的时间

df.resample('10AS').sum() 
# downsample时序数据, 频率是每10年算各列的加和, S是index从1月1日开始, 不加S则从12月30日开始
# resample的各个字符含义: A-year, M-month, W-week, D-day, H-hour, T-minute, S-second
```
除了downsample也可以upsample需要插值, 具体使用参考[官方文档](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.resample.html)




# <font color=red>绘图命令</font>

## 1. pandas内置绘图
```Python
df.plot.bar() # barplot, stacked=True, 堆叠
df.plot.barh() # 绘制水平的barplot
df.plot.hist(bins = 20) # 绘制直方图,单维度
df.plot.box() # 对每列去看一些分布outlier
df.plot.area() # 堆叠区域图
df.plot.scatter(x='a', y='b') # 散点图
df.plot.pie(subplots=True) # 绘制带图例的饼图
```

## 2. matplotlib绘图的一些设置
```Python
plt.figure(figsize=(12,8)) # 设置画布大小
plt.xticks([2,4,6], [r'a',r'b',r'c']) # 设置坐标轴刻度
plt.xlim(1,3) # 设置坐标位置
plt.title() # 标题
plt.xlabel('xxx', fontsize=18) # 绘制label
plt.text(0.8, 0.9, 'xxx', color='k', fontsize=15) # 进行注解
plt.grid(True) # 网格线 
```

## 3. seaborn绘图
```Python
    - sns.displot(x, kde=True, bins=20, rug=True, fit=stats.gamma) # histgram加密度线,样本分布情况, 拟合某些分布fit
    - sns.kdeplot # 类似于上面的,kde是每个样本用正态分布画,如果样本多,高度就高,之后再做归一化
    - sns.jointplot(x,y,data) # 绘制带有histgram以及散点图的图，两个变量
    - sns.pairplot(df) # 直接绘制各个列之间的散点图以及对应的histgram，多个变量
    - 多图绘制1
      g = sns.PairGrik(df) # 各个列混合,产出n*n个格子
      g.map_diag(sns.kdeplot) # 对角线绘制
      g.map_offdiag(sns.kdeplot, cmap='Blues_d', n_levels=20) # 绘制对角线是kde密度图其他为等高线的图
    - 多图绘制2
      g = FaceGrid(row=[..],aspect=1.5, data=)
      g.map(sns.boxplot, x, y, hue, hue_order=[], ...)
    - 多图绘制3
      g = sns.PairGrid(data, x_vars=[], y_vars=[], aspect=0.5, size=3.5)
      g.map(sns.violinplot, palette='bright') # x_vars数量*y_vars数量个子图，然后每个子图都绘制violinplot
    - 关联分析 sns.lmplot
      · sns.lmplot(x, y, data) # 散点图+线性回归,95%置信区间,适用于连续值
    - sns.residplot() # 残差图
    - sns.barplot(x,y,hue,ci=None)  # 是否打开置信区间
    - sns.stripplot(x, y, data, jitter =True) # 基于x为离散数据的,类似于散点图的boxplot
    - sns.swarmplot(x, y, data) #  蜂群图，类似于小提琴图的点版
    - sns.boxplot()
    - sns.violinplot(bw) # 属于kde以及boxplot的组合，既看了单变量分布，也看了各变量之间的差异
    - sns.violinplot(split=True， hue， inner='stick') # split将hue为两个类型的进行拼接绘制小提琴图，stick，每个样本绘制竖线
    - sns.countplot(x, data) # 绘制离散变量数量分布，类似于value_counts()，类似于barplot但是使用的统计量是数量
    - sns.pointplot(x, y, hue) # 查看离散变量x以及hue在离散变量y上的差别，使用均值，画点
    - sns.factorplot(x, y, hue, col, data, kind='swarm') # 是一种泛化的绘图函数
    - a.savefig('xx') # 进行图片存储 plt函数
```

## 4. 一些图形实例
### 4.1 barplot
```Python
import matplotlib.pyplot as plt # 绘图包的载入
% matplotlib inline # 内嵌画图, 有这个命令就可以省去plt.show()
Series.plot(kind = 'bar') # 绘制条形图
plt.xlabel('xxx') # 加x轴label
plt.ylabel('yyy') # 加y轴label
plt.title('zzz') # 加标题
```
![p1](/images/pandas_commands/p1.png)

### 4.2 scatter plot
```Python
plt.scatter(x, y, s=size, c='green') # 绘制散点图, s表明点大小, c表示点的颜色, 这两个参数都可以是list
```
![p2](/images/pandas_commands/p2.png)

### 4.3 pie chart
```Python
plt.pie([...], labels=[...], colors=[...], explode=(...), startangle=90) # 饼图的参数都是list, explode参数是为了让饼图不同类之间有空隙的参数
```
![p3](/images/pandas_commands/p3.png)

### 4.4 分组scatter plot
```Python
import seaborn as sns
sns.lmplot(x='age', y='Fare', data=df, hue='sex', fit_reg=False) # 绘制以age和Fare为xy轴的散点图, 以性别分类
```
![p4](/images/pandas_commands/p4.png)

### 4.5 histgram
```Python
plt.hist(S, bins=np.arange(0, 600, 10)) # 绘制直方图, bin可以自己定义
# 或者
a = sns.dstplot(S) # 绘制直方图, 带一条正态拟合曲线
a.set(xlabel='', ylabel='', title='') # 设定label
sns.despline() # 去掉右侧上侧的边框
```
![p5](/images/pandas_commands/p5.png)

### 4.6 correlation plot
```Python
sns.jointplot(x='', y='', data=df) #绘制带correlation散点图+histgram
```
![p6](/images/pandas_commands/p6.png)

### 4.7 pairwise散点图
```Python
sns.pairplot(df) # 多列数据绘制散点图
```
![p7](/images/pandas_commands/p7.png)

### 4.8 分类boxplot图
```Python
sns.boxplot(x='', y='', data=df, hue='') # 分类box plot, hue是类别
```
![p8](/images/pandas_commands/p8.png)

### 4.9 分类boxplot点图
```Python
sns.stripplot(x='', y='', data=df, hue='sex', jitter=True...) 
# 绘制类似于boxplot的图，只不过画的是每个box里面的点, x轴是不同数据类
```
![p9](/images/pandas_commands/p9.png)


# <font color=red>大数据知识</font>

## 1. hadoop
```bash
hadoop fs -mkdir /input/example1 # 将用于处理的文件上传到集群中，集群自动分配
hadoop fs -put text* /input/example1 # 查看文件是否放好
hadoop fs -ls /input/example1 # 集群上跑任务
hadoop jar /usr/lib/hadoop-current/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \
  -file count_mapper.py \ # 将这个文件传到集群上一会会使用
  -mapper count_mapper.py \ # 将这个文件充当mapper处理
  -file count_reducer.py \
  -reducer count_reducer.py \
  -input /input/example1 \ # 以终端的方式将文件一个个的传给命令
  -output /output/example1 
hadoop fs -getmerge /output/example1 result.txt # 运行成功之后将结果拉下来，将hadoop执行完成的结果存在/output/
```

## 2. pyspark(基于spark)

### 2.1 初始化RDD的方法
```bash
- 第一种方式直接通过内置的数据类型进行读取
  import pyspark
  form pyspark import SparkContext # 驱动
  from pyspark import SparkConf # 基本配置，内存多少，任务名称
  conf = SparkConf().setAppName("miniProject").setMaster("local[*]") # 应用名称miniProject，路径在本地
  sc = SparkContext.getOrCreate(conf) # 对上面的应用进行初始化，如果有的话直接取过来，没有的话就创建
  my_list = [1,2,3,4,5]
  rdd = sc.parallelize(my_list) # 并行化一个RDD数据，rdd是不可以直接看见，是一个对象，看不到内容
  rdd.getNumPartitions() # 存了多少份
  rdd.glom().collect() # 查看分区状况，collect是一个比较危险的命令，会把集群上的内容取到本地以列表返回，存在当前机器的内存中，可能会瞬间爆掉

- 第二种通过本地文件进行读取
  rdd = sc.textFile("file://"+os.getcwd()+'/nd.txt') # 一定要将"file//"+绝对路径的这个文件，注意这种读取是以每一行为一个元素的读取，每个元素作为一个item
  rdd = sc.wholeTextFiles("file"+cwd+"/names") # 整个文件夹的内容(多个文件)进行读取, 对/names里面所有的文本进行读取，注意~这个时候读入的内容就是以元组内容进行组织的，(文件名,文件内容)
```

### 2.2 RDD的操作
**Spark的transformations命令(非立即执行)**
```bash
- map() # 对RDD上的每个元素都进行同一操作，一进一出
- flatMap() # 对RDD中的item执行同一个操作之后得到一个list，然后以平铺的方式把list里所有的结果组成新的list，也就是一进多出
- filter() # 筛选出满足条件的item
- distinct() # 对RDD中item去重
- sample() # 从RDD中进行采样
- sortBy(keyfunc=lambda (x,y):y, ascending=False # 对RDD中的item进行排序
- takeSample(3) # 采样
- map与flatmap之间的差别
  strRDD = sc.parallelize(['hello world', 'ni hao'])
  strRDD.map(lambda x: x.split(' ')) # 返回的是[['hello', 'world'], ['ni', 'hao']]
  strRDD.flatMap(lambda x: x.split(' ')) # 返回的是[['hello', 'world', 'ni', 'hao']]
- RDD数据的Transformation可以一个接一个的串联
  def myfunc(x):
    if x%2==1:
      return 2*x
    else:
      return x
  numberRDD = sc.parallelize(range(1,11)) # 1-10
  retRDD = (numberRDD.map(myfunc).filter(lambda x: x>6).distinct())
  retRDD.collect() # 返回所有的结果
- RDD之间的操作
  rdd1.union(rdd2) # 并集, 类似于两个list相加
  rdd1.intersection(rdd2) # 交集, 类似于python中的&
  rdd1.substract(rdd2) # 差集,类似于python中的-
  rdd1.cartesian(rdd2) # 笛卡尔乘积,类似于排列组合的所有元素，python中的product
```
**Spark的action命令(立即执行)**
```bash
- collect # 危险！list形式返回
- first() # 返回第一个item
- take(n) # 返回n个item
- count() # 计算RDD中item的个数
- top(n) # 自然序排序取前n个
- reduce(n) # 做聚合
```
**pairRDD操作**
```bash
- 基本操作
  reduceByKey() # 对所有有着相同key的items执行reduce操作
  groupByKey() # 返回类似于(key, listOfValues)这种元组RDD，后面的value list是同一个key下面的
  sortByKey() # 按照key进行排序
  countByKey() # 按照key对item进行个数统计
  countByValue() # 按照value对item进行个数统计
 collectAsMap() # 与collect类似，返回的是k-v字典
- 不同pairRDD之间进行关联
  RDD1 = sc.parallelize([('a',1), ('b',2), ('c',3)])
  RDD1 = sc.parallelize([('b',20), ('c',30), ('d',40)])
  RDD1.join(RDD2).collect() # 对两个pairRDD之间利用key进行连接
  RDD1.leftOuterJoin(RDD2).collect() # 对两个pairRDD之间利用key进行左连接，RDD1作为主导
  RDD1.rightOuterJoin(RDD2).collect() # 对两个pairRDD之间利用key进行左连接，RDD2作主导
```

### 2.3 初始化Spark DataFrame的方法
```bash
- 创建SparkSession类
  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName('Python Spark SQL')\
    .config('spark.some.config.option', 'some-value')\
    .getOrCreate() # 利用spark SQL构建一个入口
- 关闭创建好的spark入口
  spark.stop() # 因为不能同时存在多个入口
- 文件的读取
  在SparkSession中可以从一个已存在的RDD或者hive表或者Spark数据源中创建一个DataFrame
  df = spark.read.csv("/path/to/your.csv") # 读入csv文件
- DataFrame操作
  df.show() # 展示数据
  df.printSchema() # 类似于Pandas里面的df.info()函数
  df.describe([...]) # 对list里面的各列进行统计
  df.select(['name', 'age']).show() # 选两列
  df.select(df['name'], df['age']+1).show() # 选取name这列同时age这列+1
  df.filter(df['age']>21).show() # 对数据进行filter
  df.groupBy('age').count().show() # 对年龄分组同时统计人数，count类似于size
  df.groupBy('age').agg(...).show() # 类似于pandas中的用法
  df.select('xx', df['xx'].cast(...).alias(...)) # 类型转换+重命名
  df.withColumn('xxx', ...) # 对列进行处理同时产生新的列xxx
  df.orderBy('xxx', ascending=False) # 对某列进行排序
  df.filter(df['value'].isNull()).count() # isNull空值处理
  df.withColumn('sex', lit('man'))) # lit是用来产生独立的数据的
  from pyspark.sql.functions import f # 有很多内置的函数,包括udf自定义函数
- 时间操作
  df.withColumn('day', dayofmonth('time')) # dayofmonth等函数的使用，这些函数也是内置在pyspark.sql.functions中的
- 类型转换
  RDD.toDF() # RDD类型转换为spark DataFrame类型
  scDF.toPandas() # 转换spark DataFrame到pandas DataFrame
  RDD = scDF.select('col1', 'col2').rdd.map(lambda r: (r[0], r[1])) # spark DataFrame转RDD
```

### 2.4 初始化Spark DataFrame的方法
```bash
# 注意DataFrame不是一个表，所以如果想用SQL的方式进行表的查询的时候需要事先构建一个表
df.createOrReplaceTemView('sqltable')
sqlDF = spark.sql('SELECT * FROM sqltable') # 使用SQL的方式对数据进行提取，spark是自己创建的，返回的结果还是DataFrame
```

<hr />



