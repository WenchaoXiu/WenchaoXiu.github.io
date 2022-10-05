---
title: Python学习笔记
tags: []
date: 2018-10-01 21:50:50
permalink:
categories: Python
description: Python基本语法总结
image:
---
<p class="description"></p>

<!-- more -->

## conda环境以及包管理
	conda list # 查看所在环境的安装的包
	conda upgrade --all # 对包进行更新
	spyder # 启动anaconda中的IDE
	conda install numpy pandas # 在某个环境下能够安装某些Python包
	conda install numpy=1.10 # 安装特定版本的包
	conda remove < package_name > # 删除包
	conda env list # 列出当前机器上创建的虚拟环境
	conda create -n env1 python=2.7 # 创建一个名为env1的环境然后在其中安装python2.7
	conda create -n env1 numpy # 创建一个名为env1的环境然后在其中安装numpy
	source activate env1 # 进入env1虚拟环境，在window上不用加source
	source deactivate # 离开环境
	conda install -n py27 ipykernel # 在虚拟环境py27下安装ipykernel
	python -m ipykernel install --user --name py27 --display-name "python2" # 在py27环境内安装ipykernel并在菜单里命名为python2
	conda env remove -n py27 # 移除py27的虚拟环境
	conda install jupyter notebook # 在conda环境中安装jupyter notebook
	%matplotlib # jupyter notebook中已交互式方式实现matplotlib的绘图
	%matplotlib inline # 不跳出，直接内嵌在web中

## 注释
	\用于一行执行太长，换行的使用
	三个单引号或者三个双引号多行注释

## 变量
	print(type(变量)) # 确定变量类型
	命名规则：驼峰命名法，下划线命名法
	import keyword
	print(keyword.kwlist) # 打印相应的关键字

## 运算符
	// # 运算取整
	/ # 正常除
	% # 取余
	** # 幂运算
	and,or,not # 逻辑运算符
	运算符优先级：幂运算，算术运算，比较运算，逻辑运算

## 数据类型（强类型）
	类型
		number
			int
			float
			complex
		boolean
		String
		List
		Tuple
		Dictionary
	类型转换
		int(a)
		str(a)
		eval(x) ？？？
		tuple(x)
		list(x)
		chr(x) # 整数转换为字符

## 输入输出操作
	f=open("xxx.txt",'r') # 这里f是一个句柄，有上限
	print(,end='') # 不换行
	f.readlines() # 读取并存储为list
	f.readline() # 逐行读取
	with # 打开文件，不用关心close
	with open() as f: # f是个句柄

## CSV处理
	import csv
	f = open("xxx.csv")
	reader = csv.reader(f) # 映射到list类型
	reader[2][4]

	with open("",'w',newline='') as f:
		writer = csv.writer(f)
		for row in data: # data 是个list
			writer.writerow(row)

	csv.Dictreader() # 转化成字典
	writer = csv.DictWriter(f, header); writer.writeheader() # 写好头行
	writer.writerows(datas) # datas是一个字典全部写进去

## 列表
	列表中的元素可以类型不同
	列表修改
		append/insert/+/extend
		lista.insert(1,[213,13,31,1341]) # 指定位置
		lista.extend(listb) # 类似于+对lista进行修改
		del lista[1] # 删除具体位置的元素
		lista.remove("xxx") # 删除第一个匹配的元素xxx
		lista.remove(["xxx","yyy"]) # 删除第一个匹配的list["xxx","yyy"]
		list.pop() # 删除指定位置的元素
	切片
		步长[::2]
		[-1::-2] # 最后一个数字开始向前隔一个取元素
	in/not in 
	排序
		lista.sort(reverse=True) # 对列表排序
	倒序
		lista.reverse() # 把列表倒过来
	count
		lista.count(1) # 统计某个元素在列表中出现的个数，没有返回0
	索引
		lista.index("xxx") # 返回某数的索引
	range
		range(0,-10,-1) # 获得0,-1...,-9


## 列表推倒式
	[i for i in range(30) if 1%2==0] # 取0-29之间偶数
	[function(i) for i in range(30) if 1%2==0] # function可以自己定义
	[ x**2 if x%2 ==0 else x**3 for x in range(10)] # 两个条件
 
## 字符串
 	str.islower() # 判断字符串是否由小写字母组成
 	"{}:{}".format("x","y") # 字符串格式化
 	"{1}:{0}".format("x","y") # 可以给定对应的参数
 	"{0:.3}".format(1/3) # 对精度进行控制的时候可以加一个冒号后面加精度
 	"{0:7}{1:7}".format("xxx","dddd") # 对于字符串来说用多长位置对其进行保留在冒号后面，对应参数是放在前面的
 	print(s.dir()) # 字符串方法

## 元组
	顺序存储相同
	不可添加修改删除
	() # 空元祖
	a = 1,2,3 # 可省略小括号

## 集合
	无序存储，不重复，可以不同类型
	s = {1,2,3,"3"}
	set("hello") # 直接把每个字符当做元素
	set() # 创建一个集合
	s.add() # 添加元素
	s.update() # 添加一个列表，集合，也可以添加多个
	remove() # 删除元素，不存在会抛异常
	discard() # 删除元素
	pop() # 随机删除
	clear() # 清空集合
	intersection或者&对集合求交
	union或者|对集合求并集
	difference(-)对集合求差集
	对称差集^ # 对称差集就是

## 字典
	key重复的时候后面的会被覆盖
	adic.get('xx') # 如果不存在xx对应的值，那么会返回None
	adic.get('xx', '100') # 如果不存在xx对应的值，那么会返回100
	keys(),values(),iteritems() # 遍历key，value，key/value
	items() # 返回时一个tuple
	for k,v in adic.items() # 遍历
	adic.clear() # 清空字典

## 函数
	global # 全局变量,命名的时候可以在命名之前加g_方便说明识别全局变量
	变量要在调用之前声明，不用定义之前声明
	list和dictionary不需要加global声明在函数内可改
	如果返回值是元组或者列表，可以用多个变量进行接收
	如果返回4个，但是只接收3个，最后一个要使用_作为占位符

## 匿名函数
	lambda [参数列表]:表达式
	例如: sum = lambda x,y:x+y # 可以有多个参数,返回只能有一个式子
	可以作为一个函数的参数赋给另外一个参数
	当然普通函数也可以作为参数传入
	a = [{'name':'ss','age':10},{'name':'yy','age':7},{'name':'zz','age':15}] # 将匿名函数作为参数传入第三方函数的参数
	a.sort(key=lambda x:x['age']) # sort方法需要传入一个key，这个key可以作为排序依据，lambda可以提取每个元素，并对元素排列
	a.sort(key=lambda x:x['age']， reverse=True) # 降序

## 面向对象基础
	类名一般是大驼峰惯例
	构造方法__init__
	类属性和实例属性(self.xx)
	self.__name # 私有属性，不能够在外部访问
	私有方法，只能类内部进行调用,__进行声明，内部调用的时候用self.__xxx()

## 异常处理
	异常发生之后后面的代码都不会执行
	程序主动raise Exception('test')可以抛出异常
	try 
	except
	捕获多个异常 except (aaa,bbb) as err:

## 包模块
	包是一个文件夹
	模块是不同的python文件
	import package.module.func()
	import package1.module1, package2.module1 # 多个模块调用
	import package1.module1 as p1m1 # 对模块进行重命名使用
	from package.module import func1 # 调用某个包某个模块的某个函数
	import sys;sys.path # 搜索模块路径，包含当前文件夹
	package.module.__file__ # 可以确定当前的模块所在的路径
	__init__.py  #在包被加载的时候，会被执行。在一个包下面可以有也可以没有__init__.py
	from package import * # 引用包下面所有的模块都加载，自动搜索是不会发生的，
		需要我们在__init__.py下进行定义才可以实现,定义的内容是__all__=["module1","module2"],
		将package下的module1和module2都加载进来
		如果想直接加载某个函数，在__init__.py里面加入from .module1 import func1, __all__=["func1"]
		这样修改完之后，可以直接from package import *，然后直接调用func1即可，不用带package.module
		restart kernel
	如果想要直接引用包,如：import package,这样的话，需要一定要有__init__.py，否则会在打印package.__file__的时候报错。
	注意import package和from package import *效果相同

## main
	if __main__=='__main__': # 这个py文件以python命令调用的时候被执行，模块导入的方式是不能执行的
	使用场景就是自己进行测试时使用，当第三方调用时不会执行

## zip
	c = list(zip(a,b))
	c = set(zip(a,b))
	c = dict(zip(a,b))
	list(zip(*c)) # 解压

## enumerate
	enumerate(list/set, start=0) # 遍历元素，start指定从哪个数字作为开始下标

## random
	import random #引入
	random.random() # 0-1
	random.uniform(1,10) # 包含1，10的浮点数
	random.randint(1,10)  # 包含1，10的整数
	random.randrange(0,20,3) # 0-20能被3整除的数
	random.choice([1,2,3]) # 随机取元素
	random.choice("qwdwq") # 随机取元素
	random.shuffle([12,3,1,4,2,3]) # 混洗
	random.sample([1,2,3,4,5], 3) # 从前面的list中选3个

## math
	import math
	dir(math) # 所有的math的功能

## Counter
	字典的继承类
	set dict list tuple 作为key
	from collection import Counter # 导入
	cnt = Counter()
	for i in [1,1,2,2,2,3]:
		cnt[i] += 1
	print cnt
	如果用key的话会报错先做第一次初始化才行
	cnt2 = Counter(alist)  #可以统计每个元素出现的次数（字符串，set,list,）
	Counter(cat=4,dogs=8,abc=-1) # 初始化counter次数，或者用dictionary构建
	Counter({'cat':4,'dogs':8,'abc':-1})
	Counter返回一个字典，如果缺失的话会返回0
	del cnt2['xx']
	.values()
	list(cnt),set(cnt),dict(cnt) # 前两个只返回key
	cnt.most_common()[0] # 对出现次数排序
	cnt.clear()
	cnt1+cnt2 # 对于key相同的value做加法，如果为0则不保留
	cnt1-cnt2 # 对于key相同的value做减法
	& # 求key相同value的最小值
	| # 求key相同value的最大值


<hr />
