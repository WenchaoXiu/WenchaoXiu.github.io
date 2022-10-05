---
title: Python数据分析1
tags: []
date: 2019-02-03 22:31:19
permalink:
categories: Data Analysis
description: 记录了Python, Numpy, Pandas, Hadoop, Spark等基本使用
image:
---
<p class="description">"Python数据分析1"与"Python数据分析2"分别是两门课的笔记，可能有相同之处</p>


<!-- more -->

# jupyter notebook使用
	1. 安装jupyter notbook扩展
		conda install jupyter notebook
		conda install -c conda-forge jupyter_contrib_nbextensions 
		pip install -i https://pypi.tuna.tsinghua.edu.cn/simple autopep8 
		pip安装加速镜像:https://www.cnblogs.com/microman/p/6107879.html

		jupyter 使用参考资料:https://zhuanlan.zhihu.com/p/33105153
		jupyter extension 参考资料:https://zhuanlan.zhihu.com/p/52890101

	2.jupyter魔法命令
		%quickref # 所有magic命令 
		%lsmagic # 打印所有magic命令

 		%config ZMQInteractiveShell.ast_node_interactivity='all'/'last_expr'
		%pprint # 打印所有结果,保证每次执行都输出,默认只输出最后一个内容
		%config ZMQInteractiveShell可以查看可选择的输出类型
		或者执行这个命令保证多输出
		from IPython.core.interactiveshell import InteractiveShell
		InteractiveShell.ast_node_interactivity = 'all'/'last_expr'

 		%%整个cell magic
		%%writefile test.py # 将cell中的命令写入文件test.py
		%%timeit代码计时
		%%bash # 在cell内可以执行bash命令
		%%writefile xx.py # 把整个cell中的内容输入到xx.py中,如果新加内容可以%%writefile -a xx.py

		%line magic命令
		%matplotline inline # 在jupyter内打印图片
		%run utils.ipynb # 执行本地的utils.ipynb文件,进行配置

		line magic和cell magic区别就在于line magic只在一行有效,cell magic在多行都有效
		具体参考:https://gispark.readthedocs.io/zh_CN/latest/pystart/jupyter_magics.html

	3. jupyter 使用linux命令
		!head -n 5 xx.txt # 直接通过jupyter行使linux命令


# python基本知识
	1.运行方式
		解释运行:直接py脚本运行
		交互运行:jupyter输入一个输出一个

	2.命名规则:
		常量大写，下划线隔开单词
		类用驼峰命名
 		del xx 删除变量xx

	3.操作优先级：
		函数调用，寻址，下标
		幂运算
		翻转运算符
		正负号
		* / %
		- + 

	4.赋值
		多重赋值:a=b=10相当于a=10,b=10
		多元赋值 a,b,c = 1,2,3
		交换赋值 a,b = b,a # 指针

	5.解包(需要拓展 参考:https://zhuanlan.zhihu.com/p/33896402?utm_source=wechat_session&utm_medium=social&s_r=0)
		l1 = [1,2,3,4,5,'6']; a,b,*c,d = l1
		l1=[1,2,3,4];b='sdaad';[*l1,*b]
		b,=[[3,4,5]] # 逗号解包 

	6.python进制及基本类型
		bin() 二进制
		oct() 八进制
		hex() 十六进制

		float('inf') 正无穷

	7.字符串
		.format() # 字符串格式化
		amount = 1234; f'请转账给我{amount:,.2f}元' # 字符串格式化
		具体参考:https://zhuanlan.zhihu.com/p/37936007
		str3.find() # 查找
		str3.replace('a','b') # a替代b
		str3.count('xx') # xx出现的个数

	8.list对象
		append和extend的区别
		sorted(alist, reverse=True,key=len/str.lower) # 对每个元素进行function,然后返回值排序

	9.元组/集合/字典
		元组不能修改
		只有一个元素的的元组需要加逗号,如(1,)

		集合是{}包含的内容,默认返回的是不重复且排序的值

		adic.get('xxx',123) # 返回key为xxx的值,如果没有的话返回123
		od = OrderedDict();od['z']='x';od['a']='b';od['c']='f' # 有序排列的字典

	10. 表达式
		exec执行声明语句,如 exec('a=5')
		eval执行表达式, 如 eval('a+5')
		三元判断表达式, 'a' if a>50 else 'b'
		列表推导式, [i+1 for i in alist if i>10], [0 if x>5 else 0 for i in range(10)] # 如果只有if，放在后面，如果if/else都有的话，放在前面
		列表迭代器: x = iter(alist); x.__next__()或者next(x),类似于链表
		生成器:函数生成,列表表达式生成????????????????????????????????????

	11. python中的异常处理
		基本用法:
			try:
				..... # 尝试执行这里
			except Exception as e:
				.... # 异常的话执行这里
			else:
				... # 如果没有异常就会执行这里
			finally:
				.... # 无论有没有错误都会执行
		py2/py3之间差别
			try
			escept Exception as e: # py3
			escept Exception, e: # py2

	12. python函数
		a,b = b,a+b,记住这个方便操作，值互换，因为python中全部都是引用，按从右向左执行，所以可以这样做
		变长参数函数:def xxx(a, *args): # xxx(....),第一个参数传给a,后面有几个参数都传给args
		匿名函数: 通过lambda执行,sum = lambda x,y:x+y
		高阶函数
			a. filter, list(filter(afunc, alist)),基于这个afunc对alist进行过滤,需要在前面加list()是python3的规则,因为是个filter的对象
			b. map,映射,list(map(afunc, alist)),对alist中所有元素依次做afunc操作
			c. reduce, 两两处理,相当于贪吃蛇,依次利用前一次操作结果和现在元素做同样的操作,如:reduce(add, alist),add自己定义的需要两个参数的求和函数,这要会不断求和,效果就是对alist各元素进行累加

	13. 文件写入
		f.writelines(['dawd\n','dada\n']) # 多行写入

	14. 目录操作
		import os
		os.path.abspath('.') # 当前绝对路径
		os.mkdir('aaaaaa') # 新建文件夹
		os.rename('file.txt', 'file2.txt') # 将file.txt重命名为file2.txt
		[i for i in os.listdir('.') if os.path.isdir(i)] # 输出当前文件夹下所有的文件夹
		[i for i in os.path.listdir('.') if os.path.isfile(i) and os.path.splitext(i)[1]=='.py'] # 判断是否是py文件

	15. 三方库
		!pip install pillow # 图像相关库,安装相关库
		from PIL import Image
		im = Image.open('xxx.png')
		print(im.format, im.size, im.mode) # 图片的一些属性
		im.show() # 图片显示,弹窗显示
		im.thumbnail(400, 100) # 对图像进行resize
		im.save('xxx.jpg', 'JPEG') # 将resize的图片存成JPEG格式

# python正则表达式
	1.正则表达式的match与search区别
		https://segmentfault.com/a/1190000006736033

	2.贪婪匹配与非贪婪匹配的区别
		https://segmentfault.com/a/1190000002640851
		https://blog.csdn.net/lxcnn/article/details/4756030

	3. 练习网站
		https://alf.nu/RegexGolf 一个正则表达式练习网站
		https://regexr.com/ 验证网站

	4. 单字符匹配
		. # 匹配出点换行符之外的任意字符
		\. # 匹配单个.字符
		[abd] # 匹配a/b/d单个字符
		\d # 匹配数字, 相当于[1,2,3,4,5,6,7,8,9]
		\D # 所有非字符
		\s # 空白符,空格 tab等等
		\S # 所有非空格
		\w # a-z,A-Z,0-9,_
		\W # 除了 a-z,A-Z,0-9

	5. 数量词用来多匹配
		m{2} # 表示匹配两个m
		m{2,4} # 表示匹配2/3/4个m,贪婪匹配
		m* # 0个或者更多个,贪婪匹配
		m+ # 1个或者更多个,贪婪匹配
		m? # 0个或者1个
		^xx # 文本开头是xx进行匹配
		xxx$ # 对结尾进行匹配
		(re)su(lt) # group

	6. python中的正则表达式步骤
		写一个文本pattern
		进行匹配
		对匹配的文本进行后续操作
		例子:
			import re
			pattern = re.compile(r'hello.*\!') # hello后面有若干个字符串直到有!
			match = pattern.match('hello, xxx! how are you?') # 对文本进行匹配
			if match: # 是否匹配
				print match.group() # 如果匹配上了返回相应的匹配到的部分

	7. 使用实例
		import re
		re.compile(r"""
		\d+ # 数字部分
		\. # 小数点
		\d # 小数部分
		""", re.X)
		这种模式下可以写注解
		re.compile(r"\d+\.\d") # 与这个模式结果一样

	8.一些命令
		match # 一次匹配结果,从头匹配,开头没有就匹配不上了
		search # 所有匹配到的
		findall # search返回第一个匹配的结果,findall会返回所有的结果
		m=re.match()
		m.string # 匹配的字符串
		m.group(1,2) # 匹配1和2处字符串

	9. 替换和分割
	 	split也可以使用正则表达式进行分割
			p = re.compile(r'\d+')
			p.split('adwdwad1dawwd23dwadw') # 字符串复杂分割
		sub # 用来替换
			p = re.compile(r'(\w+) (\w+)')
			p.sub(r'\2 \1', s) # 匹配字符串并且在匹配到的字符串处进行前后颠倒 
		subn # 和sub类似,只不过除了返回替换的远足之外,还返回相应的替换次数,可以p.subn(afunc, s), afunc可以自己定义

# numpy的使用
	1. 优点:向量化数据操作比for循环,速度大大加强，numpy array比list好的地方在于切片

	2. array属性
		np.random.random((2,2)) # 0-1随机数
		np.random.randint(1,10,(3,3)) # 随机整数
		array.shape, array.dtype # numpy两个属性
		array.astype(np.float64) # 类型转换

	3. array切片操作
		a[0,1] # 第一个维度为0,第二个维度1,第三个维度全选,类似于a[0,1,:]
		a[a>2] # boolean indexing, 利用broadcasting进行判断, 之后可以作为index进行数据的提取
		a[a>2]=0 # 也可以对满足条件的元素进行赋值

	4. array数学运算
		broadcasting, 对不匹配的数据在高维上进行扩展,在取最小公倍数
		np.sum(array) # 统计运算
		np.dot # 矩阵乘法,点乘
		np.multiply  # 逐个元素乘法,对应相乘

# Pandas的使用
## Series
		1. 基本概念
			pd.__version__ # 查看版本
			pd.Series # 可以使用不同类型,和list区别在于有index, 可以指定index

		2. Series构建
			pd.Series([1,2,3], index=['a', 'b', 'c'])
			pd.Series({...}, name="xxx") # 通过对dictionary进行构建pandas, 给Series赋予名字

		3. 切片
			aseries[[1,4,3]]; aseries[1:]; aseries[:-1]  # 数字下标切片,即使index不是数字也ok

		4. 运算规则
			series的相加是根据index对应相加的

		5. 取值
			数学运算也是broadcasting方式
			'xxx' in aseries # 判断xxx是否在aseries的index中
			aseries.get('xxx', 0) # 类似于字典
			aseries[aseries<20] # boolean index也可以
			aseries.median() # 除去缺失值之后进行统计运算
			aseries['xxx'] = 1000 # 对aseries['xxx']重新赋值
			np.square(aseries) # 对每个运算进行计算平方
## DataFrame
		1. 基本概念
			一组Series集合在一起

		2. DataFrame的构建
			- pd.DataFrame({'a':[1,2,3], 'b':[1,4,3]}, columns = ['b', 'a'], index = ['one', 'two', 'three']) # 构建DF, 指定列名以及index名
			- pd.DataFrame([{'a':100,'b':200}, {'a':200, 'b':300}], index=['one', 'two']) # 按照一行一行构建DF
			- pd.DataFrame({'a':seriesa, 'b':seriesb} # 记住按照index对齐, 缺失值直接Nan填充

		3. 元素的提取以及增加及逻辑操作及转置
			- aDF['xxx']/aDF.xxx # 取出来的是一个Series
			- aDF[['xxx']] # 取出来的是一个DF
			- aDF.loc(['a','b'],['c','d']) # 取对应的数据
			- aDF.loc[:, 'newcol'] = 2000 # 如果没有newcol那么就新加一列
			- aDF.loc[(aDF['a']>10) & (aDF['b']<100), :] # 也可以给条件进行筛选,& | ~进行逻辑运算
			- aDF.T # 进行转置

		4. 数据读入以及基本信息以及删除
			- pd.read_csv(path, sep='\t', index_col=''/int, usecols=[...], header=0, parse_dates=[0]/['Date']) # 读文件，第一列作为日期型，日期型处理参照: http://hshsh.me/post/2016-04-12-python-pandas-notes-01/
			- aDF.to_csv('xxx.csv', sep='\t', index=True, header=True) # 写文件
			- aDF.describe(include=[np.float64...]) / aDF.info() # 对数据进行统计，查看缺失值
			- aDF.shape
			- aDF.isnull() # 判断是是否为空
			- aDF[aDF['xxx'].isnull(), :] = 10 # 对空值赋值
			- aDF.notnull() # 查看是否有值
			- aDF.drop(['one', 'two'], axis=0) # 对index为one和two的两行进行删除, axis=1删除列

		5. 数据分组聚合
			- aDF.groupby('name', sort=False).sum() # 对DF进行聚合操作,同时对相应聚合的列进行排序,然后计算其他值的和
			- groupbyname=aDF.groupby('name'); groupbyname.groups; len(groupbyname) # 得到对应的各个组别包含的index, 并且可以获取对应的group长度
			- aDF.groupby('name').agg([np.sum, np.mean, np.std]) # 对不同类别的数据进行各类运算, 每个name对应三列分别是分组之后np.sum, np.mean, np.std计算
			- aDF.groupby('name').agg(['sum', 'median', 'mean']) # 和上面的作用相同
			- aDF.groupby('name').agg(['a':np.sum, 'b':median, 'c':np.mean]) # 对不同列进行不同操作
			- aDF.groupby(['name', 'year']).sum()/mean()/median()/describe() # 多组分类
			- aDF.groupby(['name', 'year']).size() # 多组分类, 每一组有多少个记录
			- 提取group类别名称以及类别对应的数据行
				for name,group in groupbyname:
					print(name) # 类别名称
					print(group) # 名称对应的数据行
				groupbyname.get_group('jason') # 可以得到对应组别的数据行,DF格式

		6. transform/apply/filter 数据变换
			transfrom可以对分组进行变换, apply对整个DF进行分类,filter对分组进行判断
			- aDF['Date'].dt.dayofweek # 可以得到对应的日期中的第几天
			- aDF.groupby(aDF.index.year).mean() # 可以对相应的日期型的年进行分组聚合
			- aDF.groupby(aDF.index.year).transform(lambda x: (x-x.mean())/x.std()) # 对每一年的数据求均值以及标准差,并对每个数据进行操作,之所以没以每年为单位进行展示主要是跟function有关,因为之前的是mean之类的
			- aDF.groupby(aDF.index.year).apply(lambda x: (x-x.mean())/x.std()) # 可以起到相同的效果
			- aDF.loc[:,'new'] = aDF['xxx'].apply(afunc) # 可以对xxx这一列进行操作按照afunc进行操作,然后创建新的列
			- aSer = pd.Series([1,1,2,2,2,3,3,4,5,5]); sSer.groupby(sSer).filter(lambda x:x.sum()>4) # 对ser进行过滤,留下那些和大于4的类别

		7. 表格的拼接与合并(concat/append/merge/join)
			- df1.append(df2, sort=False, ignore_index=True) # 追加在行上,同时忽略原先df1和df2的index,合并为新的index
			- df1.append([df2, df3])  # 也可以追加两个DF, 参考: https://zhuanlan.zhihu.com/p/38184619
			- pd.concat([df1.set_index('a'), df2.set_index('a')], sort=False, axis=1, join='inner') # 和上述利用merge在a字段上进行内连接的效果类似,因为concat是基于index进行连接的,merge可以不基于index,指定字段
			- pd.concat([df1, df2, df3], keys=['a', 'b', 'c'], axis=0, join='outer', sort=False) #列对齐的方式对行进行拼接,缺少值则补充为None,可以对拼接的每个df进行key的命名,axis=1的时候行对齐列拼接; join指定连接方式,outer表示外连接,inner表示内连接,sort是否对合并的数据进行排序
			- merge # 基于某个字段进行连接,之前的append和concat都是在行上或者列上进行连接的,merge类似于SQL里面的连接,可以指定某个字段或某几个字段,体现在on上,on接list就是多个字段为key
			- pd.merge(df1, df4, on='city', how='outer'/'inner'/'left'/'right') # 基于两个表中的city字段进行表格的连接,把其他的列进行combine到一起,不指定on的话就会找字段相同的那个进行拼接,注意concat是基于index进行拼接的
			- pd.merge(df1, df2, how='inner', left_index=True, right_on='id') # 对数据进行merge,左表以index作为连接关键字,右表用id作为关键字

		8. Case study流程
			- aDF.groupby('a').agg({'xx':'count', 'ccc':np.mean}).rename(columns={'old':'new'}) # 查看以a为分组xx的数量,以及ccc的均值,对old进行重新的命名叫做new
			- pd.to_numeric(aDF.loc[:, 'xxx'])  #转换成数字
			- pd.merge(df1, df2, how='inner', left_index=True, right_on='id') # 对数据进行merge,左表以index作为连接关键字,右表用id作为关键字
			- aDF.sort_values(by=['a','b'], ascending=False)  # 进行排序依据这两个字段ab,倒序排列
			- !head -n 5 xx.txt # 直接通过jupyter行使linux命令
			- pd.read_csv('xxx', encoding='latin-1',sep='\t') # 对数据进行编码指定
			- aDF.dropna(how = 'all', axis=0) # 对行进行操作,如果一行里面全都是na那么就删除,如果要一列里面全是na就删除,那么axis=1
			- aDF.loc[:,'xx'].fillna(aSeires) # 对数据进行填充,根据aSeires的index和aDF的index进行填充
			- aDF.plot(grid=True) # 对折线图绘制,同时加上网格
			- pd.DataFrame({'xx':aDF['xx'].bfill(), 'yy':aDF['yy'].bfill(), }) # 对数据填充,backfill,对前面空值用后面有值处进行填充
			- aDF[aDF.index >= aDF.xx.first_valid_index()] # 找到xx列最初非nan值的行,然后取出这行之后的所有行
			- aDF.resample('M').last() # reample是对数据进行重新采样，一月份为单位，类似于groupby统计一个月的数据，.last是统计这个月所在group的最后一个记录

		8. 链家Case study流程
			- pd.to_datetime() # 日期类型转换
			- df.drop(droplist, inplace=True, axis=1) # 删除一些列
			- aDF.describe(include='all') # 字符串变量也会同时统计
			- aDF.sort_values(by = 'xxx').tail() # 找出更新最晚的20套,但是有可能同一天超过20套
			- 如果对数据进行处理发现转换未果可能是因为数据有缺失,做异常处理,缺失值作为Nan
			- aDF.nsmallest(columns='age', n=20) # 取出年龄最小的20个数据
			- groupby().agg() 之后一般会使用reset_index() 对数据进行归置然后再进行操作,ascending=False
			- adf.value_counts(normalize=True) # 默认是按照value进行排序的
			- aDF.apply(lambda x: 'xxx' in x) # 筛选出xxx在某列的值中与否,返回Ture, False，正则表达式的字符串匹配
			- 可以定义正则表达式对文本信息进行提取
				def get_info(s, pattern, n):
					result = re.search(pattern, s)
					if result:
						return result.group(n)
					else:
						return ''
			- .astype(int) # 转换pd类型
			- help(pd.Series.value_counts) # 打印帮助文档

# python绘图
	1. pandas 绘图
		- pd.date_range('2018/12/28', periods=10) # 产生日期格式, 以2018/12/28为起始产生以天为单位的日期时间list
		- pandas绘图需要把横坐标作为index,之后再画图
		- 折线图绘制需要注意各列幅度，否则数值不明显
		- df.plot.bar() # barplot, stacked=True, 堆叠
		- df.plot.barh() # 绘制水平的barplot
		- df.plot.hist(bins = 20) # 绘制直方图,单维度
		- df.plot.box() # 对每列去看一些分布outlier
		- df.plot.area # 堆叠区域图
		- df.plot.scatter(x='a', y='b') # 散点图
		- df.plot.pie(subplots=True) # 绘制带图例的饼图

	2. matplotlib 绘图
		- plt.rcParams['figure.figsize'] = (12,8) / plt.figure(figsize=(12,8)) # 设置画布大小
		- ax = plt.plot(x,y,color='green', linewidth='-', marker='./*/x', label=r'$y=cos{x}$'/r'$y=sin{x}$'/r'$y=\sqrt{x}$') # 绘图
		- ax.spines['right'].set_color('none') # 去掉右边的边框
		- ax.xaxis.set_ticks_position('bottem') # ??????????????
		- plt.xticks([2,4,6], [r'a',r'b',r'c']) # 设置坐标轴刻度
		- ax.spines['bottem'].set_position('data', 0)  # 设置坐标轴从0开始
		- plt.xlim(1,3) # 设置坐标位置
		- plt.title() # 标题
		- plt.xlabel(r'xxx', fontsize=18, labelpad=12.5) # 绘制label, r值的是不转义的,$$值的是markdown格式
		- plt.text(0.8, 0.9, r'$$', color='k', fontsize=15) # 进行注解
		- plt.scatter([8], [8], 50, color='m') # 在某个位置,点有多大,颜色是什么
		- plt.annotate(r'$xxx$', xy=(8,8), xytext=(8.2, 8.2), fontsize=16, color='m', arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.1', color='m')) # 对某个点进行注解, 进行加箭头等等
		- plt.grid(True) # 网格线 
		- plt.plot(x, y) # xy应为np array,如果是pandas那么可以通过values进行取值转换

	3. matplotlib 绘图case
		- 文件解压
			x = zipfile.ZipFile(xxx, 'r') # 解压文件夹
			x.extractall('xxxdir') # 解压到某个文件夹下
			x.close() # 记得关闭
		- matplotlib.rc('figure', figsize=(14,7)) # 设置一下图片尺寸
		- matplotlib.rc('font', size=14) # 设置字体
		- matplotlib.rc('axes.spines', top=False, right=False) # 设置边线
		- matplotlib.rc('axes', grid=False) # 设置网格
		- matplotlib.rc('axes', facecolor='white') # 设置颜色
		- fig,ax含义
			fig,ax = plt.subplots() # 创建绘图对象之后对ax进行操作，相当于先fig=plt.figure()再ax=fig.add_subplot(1,1,1)
			https://blog.csdn.net/htuhxf/article/details/82986440
		- ax.fill_between(x, low, upper, alpha=) # 对回归进行置信度绘制
		- ax2 = ax1.twinx() # 共享同一个x轴
		- ax2.spines['right'].set_visible(True) # 对右侧坐标轴进行设置,得到相应的图
		- 图的使用
			关联分析:散点图,曲线图,置信区间曲线图,双坐标曲线图
			分布分析:堆叠直方图, 密度图
			组间分析:柱状图(带errorbar),boxplot,这个需要多看看,

	4. seaborn 绘图
		- 引入seaborn的同时也要引入matplotlib因为,是底层
		- 颜色设置
			sns.set(color_codes=True) # 一些集成的颜色
			https://seaborn.pydata.org/tutorial/color_palettes.html
		- sns.displot(x, kde=True, bins=20, rug=True, fit=stats.gamma) # histgram加密度线,样本分布情况, 拟合某些分布fit
		- sns.kdeplot # 类似于上面的,kde是每个样本用正态分布画,如果样本多,高度就高,之后再做归一化
		- sns.jointplot(x,y,data) # 绘制带有histgram以及散点图的图，两个变量
		- sns.pairplot(df) # 直接绘制各个列之间的散点图以及对应的histgram，多个变量
		- scatter plot的密度版
			with sns.axes_style('ticks'):
				sns.jointplot(x,y,data, kind='hex'/'kde',color='m') #相当于对点很多的时候,六角箱图就能体现出点的多少,kde是等高线,密度联合分布
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
			· sns.lmplot(x, y, data, x_jitter=0.08) # 左右抖动, 点如果离得近,会把点左右抖动开,适用于离散值
			· sns.lmplot(x, y, data, x_estimator=np.mean, ci=95, scatter_kws={'s':80}, order=2, robust=True) # 对于离散值还可以这样操作,先求均值和95置信区间,之后再进行拟合, scatter_kws对点进行操作,order是说对数据点进行二次方的分布,而不是线性分布,robust打开的作用是踢除异常点,然后再进行绘制图
			· sns.lmplot(x, y, data, x_estimator=np.mean, ci=95, scatter_kws={'s':80}, order=1, robust=True, logistic=True) # 相当于是说对二值化的数据进行logistic回归拟合,sigmoid拟合
			· sns.lmplot(x, y, data, hue, col, row, col_wrap, aspect=0.5) # 散点图.线性回归,95%置信区间,适用于连续值,hue进行分组类似于pandas里面的groupby, hue变量一定是个离散变量, col也可以加一个变量,可以把图分成多列,row可以多行,如果row,col以及hue都指定,那么相当于在pandas里面groupby三个内容,col_wrap用于之指定每个col中的绘图数量
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

# 大数据基本知识
	- 基本软件
		hadoop HDFS（hadoop distributed file systems）# 存储
		Map-reduce # 计算框架，用于进行数据分析，思想
		Hive # 数据库，类似于mysql
		strom # 时延小实时的，吞吐量小
		Flink # 吞吐量大
	- 重点学习，批处理，
		hadoop的HDFS的相关知识和操作命令，Hadoop map-reduce的计算核心思想，
		spark的RDD(弹性分布式数据集)核心，transformation和action，基于DataFrame的操作，Spark SQL

# Hadoop数据
	- hadoop介绍
		namenode # 存储相应的datanode的信息
		datanode # 存储数据，有冗余的存储，多个block，通常每个64M
		MapReduce  # 在每个DataNode上做操作(map),最后比较DataNode上比较好的数据(reduce)
	- hadoop集群命令
		hadoop fs -ls / # 显示集群/下的所有信息
		hadoop fs -lsr / # 递归显示所有文件夹和子文件夹
		hadoop fs -mkdir /user/hadoop # 创建/user/hadoop文件夹
		hadoop fs -put a.txt /user/hadoop # 把a.txt放到集群上的/user/hadoop文件夹下
		hadoop fs -get /user/hadoop/a.txt / # 把集群上的a.txt拉到本地上来
		hadoop fs -cp src dst # 在集群上复制文件src和dst都是集群上的文件夹
		hadoop fs -mv src dst # 在集群上进行文件移动
		hadoop fs -cat /user/hadoop/a.txt # 查看集群上的文件
		hadoop fs -rm /user/hadoop/a.txt # 删除集群上文件
		hadoop fs -rmr /user/hadoop/ # 删除集群该目录上的所有文件
		hadoop fs -copyFromLocal localsrc dst # 与hadoop fs -put的功能类似
		hadoop fs -moveFromLocal localsrc dst # 将本地文件上传到hdfs并且删除本地文件

# map-reducer
	可以利用bash shell命令进行本地测试
	- 介绍
	以词频统计来说，因为hdfs只能一行一行的进行读取，所以可以利用两个脚本分别作为mapper和reducer来进行词频统计，map用于将文本中每一个词都进行输出，以键值对这种形式，当然因为每次只统计单个词所以，输出的词频都为1。经历了mapper以后之后会经历sort排序，这个是内部完成的。最后会进行reducer步骤，当然也要通过一个python脚本进行执行，这个python脚本读取的也是上一行的标准输出，每次读一个键值对，因为键值对已经排好序了，所以直接基于排序进行逐行读取统计，如果遇到新词就把原来的词进行输出，同时当前的词进行更新。这样就完成了map-sort-reduce的脚本。
	-至于具体执行
		# 本地测试结果
		head -n 200 text1.txt|python count_mapper.py|sort|python count_reducer.py 
		# 创建文件夹，如果存在删除，hadoop fs -rmr /input/example1
		hadoop fs -mkdir /input/example1 
		# 将用于处理的文件上传到集群中，集群自动分配
		hadoop fs -put text* /input/example1
		# 查看文件是否放好
		hadoop fs -ls /input/example1
		# 集群上跑任务
		hadoop jar /usr/lib/hadoop-current/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \
		-file count_mapper.py \ # 将这个文件传到集群上一会会使用
		-mapper count_mapper.py \ # 将这个文件充当mapper处理
		-file count_reducer.py \
		-reducer count_reducer.py \
		-input /input/example1 \ # 以终端的方式将文件一个个的传给命令
		-output /output/example1 
		# 运行成功之后将结果拉下来
		hadoop fs -getmerge /output/example1 result.txt # 将hadoop执行完成的结果存在/output/example1这里面，然后将里面的子文件全部都合并，存到本地命名为result.txt文件

# spark核心概念与操作
	1. 优点
		可以实现map-reduce，支持数据挖掘，图运算，流式计算(实时)，SQL的多种框架(批量造特征)
		基于内存，试用于迭代多次的运算
		能与Hadoop的HDFS文件系统相结合，能运行在YARN上
	2. 一些概念
		RDD时如何完成transformation和action的、高版本的Spark的DataFrame的操作
		SparkContext(驱动程序)，ClusterManager(集全资源管理器)和Excutor(任务执行进程)
		所有的spark上的操作都会转化为，RDD弹性分布式的数据集上的突然transformation和action
		RDD可以类似于numpy的array或者pandas的series或者python的list
	3. 初始化RDD的方法
		- 第一种方式直接通过内置的数据类型进行读取
			import pyspark
			form pyspark import SparkContext # 驱动
			from pyspark import SparkConf # 基本配置，内存多少，任务名称
			conf = SparkConf().setAppName("miniProject").setMaster("local[*]") # 应用名称miniProject，路径在本地
			sc = SparkContext.getOrCreate(conf) # 对上面的应用进行初始化，如果有的话直接取过来，没有的话就创建
			my_list = [1,2,3,4,5]
			rdd = sc.parallelize(my_list) # 并行化一个RDD数据，rdd是不可以直接看见，是一个对象，看不到内容，因为数据被分发到各个位置。注意python的list或者numpy array或者pandas的Series/DataFrame都可以转化成RDD
			rdd.getNumPartitions() # 存了多少份
			rdd.glom().collect() # 查看分区状况，collect是一个比较危险的命令，会把集群上的内容取到本地以列表返回，存在当前机器的内存中，可能会瞬间爆掉
		- 第二种通过本地文件进行读取
			# 存储在本地服务器的文件进行读取(假设本地读取的文件为:"/name/example1.txt")
				# 单个文件进行读取
				import os
				cwd = os.getcwd()
				rdd = sc.textFile("file://"+os.getcwd()+'/nd.txt') # 一定要将"file//"+绝对路径的这个文件，注意这种读取是以每一行为一个元素的读取，每个元素作为一个item
				rdd.first() # 对读入文件进行查看，第一行
				# 整个文件夹的内容(多个文件)进行读取
				rdd = sc.wholeTextFiles("file"+cwd+"/names") # 对/names里面所有的文本进行读取，注意~这个时候读入的内容就是以元组内容进行组织的，(文件名,文件内容)
				rdd.first() # 第一个元素就是文件名+文件内容
				rdd.count() # 统计RDD文件的行数
		- 其余初始化方式
			HDFS上的文件
			Hive中的数据库与表
			Spark SQL得到的结果
	4. RDD的transformations和actions
		- map() 对RDD上的每个元素都进行同一操作，一进一出
		- flatMap() 对RDD中的item执行同一个操作之后得到一个list，然后以平铺的方式把list里所有的结果组成新的list，也就是一进多出
		- filter() 筛选出满足条件的item
		- distinct() 对RDD中item去重
		- sample() 从RDD中进行采样
		- sortBy() 对RDD中的item进行排序
		- collect() 如果直接获取相应的结果可以使用rdd.collect()对结果组织成list进行获取
		- 例子
			numberRDD = sc.parallelize(range(1, 11))
			print numberRDD.collect()
			squaresRDD = numberRDD.map(lambda x:x**2) # 对所有数据进行平方，可以自己定义个函数
			print squaresRDD.collect()
			filteredRDD = numberRDD.filter(lambda x:x%2==0) # 对所有数据进行筛选，返回True对应的元素
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
	5. Spark的核心是惰性计算，各种transformation不会立即生效，不像pandas的操作一样，他会重新组织你的transformation，这样可以避免一系列的中间结果。
	6. Spark的唤醒机制，就是各种action命令
		- collect # 危险！list形式返回
		- first() # 返回第一个item
		- take(n) # 返回n个item
		- count() # 计算RDD中item的个数
		- top(n) # 自然序排序取前n个
		- reduce(n) # 做聚合
		- 例子
			rdd = sc.parallelize(range(1,11))
			rdd.reduce(lambda x,y : x+y) # 注意！这个会立即执行，类似于python高阶函数reduce
		sc.stop() # 停止sparkContext
	7. Spark中的cache()命令，如果一遍一遍计算的话，需要开销，所以可以通过cache命令将他们存储在内存中
		- 例子
			import numpy as np
			numberRDD = sc.parallelize(np.linspace(1.0, 10.0, 10))
			squareRDD = numberRDD.map(lambda x: x**2)
			squareRDD.cache() # 表明如果squareRDD这个对象第一次action了之后，就直接将其写入缓存，将中间结果存一下，这样后面计算的时候不用再激活，就不会很耗时
			avg = squareRDD.reduce(lambda x,y:x+y)/squareRDD.count() # 计算平均值，如果不写cache的话会导致action执行两次，会比较耗时
	8. 复杂结构的transformation和action
		- 介绍
			基本的存储结构，用来进行统计词频，主要是以(key,value)进行组织的，把这种形式叫做pair RDDS
			记住依然以list形式进行组织，只不过list的每个元素长成(key,value)这种形式
			针对这种数据结构定义了一些transformation和action
			reduceByKey() # 对所有有着相同key的items执行reduce操作
			groupByKey() # 返回类似于(key, listOfValues)这种元组RDD，后面的value list是同一个key下面的
			sortByKey() # 按照key进行排序
			countByKey() # 按照key对item进行个数统计
			collectAsMap() # 与collect类似，返回的是k-v字典
		- 例子
			rdd = sc.parallelize(['ni hao', 'wo bu hao', 'na ni ne'])	
			retRDD = (rdd.flatmap(lambda x:x.split(' '))
					.map(lambda x:x.lower())
					.map(lambda x:(x,1))
					.reduceByKey(lambda x,y:x+y)) # 类似于groupby+aggregate
			retRDD.collect() # 对上述词频统计的结果进行返回
			retRDD.collectAsMap() # 对词频统计结果以字典形式进行返回
			retRDD.sortBy(keyfunc = lambda (k,v):v, ascending=False).take(2) # 找到词频统计中次数最高的两个
		- 不同pairRDD之间进行关联
			RDD1 = sc.parallelize([('a',1), ('b',2), ('c',3)])
			RDD1 = sc.parallelize([('b',20), ('c',30), ('d',40)])
			RDD1.join(RDD2).collect() # 对两个pairRDD之间利用key进行连接
			RDD1.leftOuterJoin(RDD2).collect() # 对两个pairRDD之间利用key进行左连接，RDD1作为主导
			RDD1.rightOuterJoin(RDD2).collect() # 对两个pairRDD之间利用key进行左连接，RDD2作主导
			- cogroup
				x = sc.parallelize([('a',1), ('b',2), ('c',3)])
				y = sc.parallelize([('b',20), ('c',30), ('d',40)])
				x.cogroup(y).collect() # 以key进行联合，有多少个key接结合多少，类似于outerjoin，以(key,(v1,v2...))返回
				x.cogroup(y).map(lambda (x,(y,z)):(x,(list(y),list(z)))).collect() # 因为直接collect返回的是对象不是具体的内容，所以需要自己进行转换

# Spark DataFrame
	- 介绍
		spark SQL上可以构建类似于Pandas DataFrame的结构
		Spark SQL的功能入口点是SparkSession类，如果要创建一个SparkSession，使用SparkSession.builder()即可
	- 创建SparkSession类
		from pyspark.sql import SparkSession
		spark = SparkSession.builder.appName('Python Spark SQL').config('spark.some.config.option', 'some-value').getOrCreate() # 利用spark SQL构建一个入口
	- 创建DataFrame类
		在SparkSession中可以从一个已存在的RDD或者hive表或者Spark数据源中创建一个DataFrame
		df = spark.read.json("data/people.json") # jason格式就是类似于字典的形式
		df.show() # 展示数据
	- DataFrame操作
		df.printSchema() # 类似于Pandas里面的df.info()函数
		df.select('name').show() # 选一类
		df.select(['name', 'age']).show() # 选两类
		df.select(df['name'], df['age']+1).show() # 选取name这列同时age这列+1
		df.filter(df['age']>21).show() # 对数据进行filter
		df.groupBy('age').count().show() # 对年龄分组同时统计人数，count类似于size
	- Spark SQL
		注意DataFrame不是一个表，所以如果想用SQL的方式进行表的查询的时候需要事先构建一个表
		df.createOrReplaceTemView('people')
		sqlDF = spark.sql('SELECT * FROM people') # 使用SQL的方式对数据进行提取，spark是自己创建的，返回的结果还是DataFrame
		sqlDF.show()
		sqlDF.rdd # 就是一个pyspark.rdd.RDD对象，因此可以通过sqlDF.rdd.first()对rdd对象进行访问
	- rdd构建spark DataFrame
		sparkDF = spark.createDataFrame(sc.parallelize([1,2,3,4])) # 构建spark DataFrame对象
		sparkDF.createOrReplaceTemView('people') # 从spark DataFrame对象构建SQL的表
	- StructField和StructType


# 一些参考资料
[pyspark package](https://spark.apache.org/docs/2.1.0/api/python/pyspark.html)

[pyspark sql module(DataFrame)](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader)

[pyspark dataframe基础1](https://zhuanlan.zhihu.com/p/34901683)
[pyspark dataframe基础2](https://blog.csdn.net/sinat_26917383/article/details/80500349#_30)

[Python机器学习](https://cloud.tencent.com/developer/article/1366850)

[HDFS常用命令](http://blog.sanyuehua.net/2017/11/01/Hadoop-HDFS/)

[pyspark列类型状换](https://stackoverflow.com/questions/40478018/pyspark-dataframe-convert-multiple-columns-to-float)

[pyspark 小数据位数保留](https://stackoverflow.com/questions/48832493/how-to-set-display-precision-in-pyspark-dataframe-show)

[pyspark RDD数据基本使用介绍](https://blog.csdn.net/cymy001/article/details/78483723)

[pyspark DataFrame年月日提取](https://stackoverflow.com/questions/30949202/spark-dataframe-timestamptype-how-to-get-year-month-day-values-from-field)

[数据科学速查表](https://blog.csdn.net/sunkun2013/article/details/73250874)

[正则表达式练习网站](https://regexr.com/)

[一个机器学习互动网站](https://okai.brown.edu/zh/chapter1.html)

[markdown使用](http://xianbai.me/learn-md/article/syntax/paragraphs-and-line-breaks.html)

[python面向对象编程1](http://www.runoob.com/python/python-object.html)
[python面向对象编程2](http://yangcongchufang.com/%E9%AB%98%E7%BA%A7python%E7%BC%96%E7%A8%8B%E5%9F%BA%E7%A1%80/python-object-class.html)

[python高阶知识(可迭代对象,迭代器,生成器)](https://foofish.net/iterators-vs-generators.html)

[conda使用](https://www.linqingmaoer.cn/?p=201)

[通过修改对象的__eq__方法改变==的比较方式](https://zhuanlan.zhihu.com/p/26488074)

[机器学习基础数学知识](https://zhuanlan.zhihu.com/p/25197792)


<!-- 
linux free用来查看内存

JData相关
	https://jdata.jd.com/html/detail.html?id=8
	https://jdata.jd.com/html/detail.html?id=1
	https://github.com/daoliker/JData
	https://github.com/hecongqing/2017-jdata-competition

学员笔记
	- https://shimo.im/docs/di4oZVafbL47taey/read
 -->




<hr />
