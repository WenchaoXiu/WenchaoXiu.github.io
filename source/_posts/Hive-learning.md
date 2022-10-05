---
title: HIVE命令学习
tags: []
date: 2020-03-04 14:03:49
permalink:
categories: HIVE
description: 使用HIVE进行大体量数据的数据分析学习笔记
image:
---
<p class="description">该学习笔记主要基于b站尚硅谷中HIVE教程</p>

<!-- <img src="https://" alt="" style="width:100%" />
 -->

<!-- more -->

## <font color=red>Hive简介</font>
### Hive基本概念
1. Hive是一个分析框架，区别于Hadoop，Hadoop是存储及分析框架，存储对应分布式管理系统HDFS，分析框架是MapReduce。是一个基于Hadoop的**数据仓库工具**，将结构化的业务数据映射为一张表，提供类SQL查询功能。
2. 本质是，将HQL转化为MapReduce程序，类似于Hadoop的一个客户端。
    a. Hive处理的数据在HDFS中
    b. 底层实现是MapReduce
    c. 执行程序运行在Yarn上

### Hive优缺点
#### 优点
1. 类SQL，简单，易上手
2. 适用于大数据，小数据无优势，因为延迟高
3. 可以自定义函数，扩展性好

#### 缺点
1. 延迟性比较高，实时性要求不高的场合，离线场景
2. 迭代式计算无法表达，及多个MR（MapReduce）串联较难
3. 数据挖掘方面不擅长，特指ML（Machine Learning）算法之类的
4. 效率低，基于Hadoop来的

### Hive架构
其中Meta store是存储相应的表与文件的相关信息的，包括文件大小，文件位置等等信息，存储在数据库中，默认是derby，后面可以换成MySQL。换句话说通过HQL语句转化为底层的MR语句，查询的原数据存在HDFS，元数据(meta data)处于数据库中。
<div style="width: 300px; margin: auto">![Hive架构](Hive架构.png)</div>


### Hive与数据库的比较
1. **查询语言**：类似
2. **存储位置**：Hive是Hadoop的HDFS中的，SQL是存储在本地中的。
3. **数据更新**：SQL数据库需要经常修改，而Hive则*读多写少*
4. **索引**：Hive没有索引，因为数据比较大，建索引意义不大，不适合在线数据查询
5. **执行**：Hive通过MR实现的，而数据库有自己的执行引擎
6. **执行延迟**：Hive由于没有索引需要扫描表，延迟高，启动慢
7. **可扩展性**：Hive基于Hadoop，所以存储扩展和计算扩展都较高，MySQL也可扩展，但是相对来说扩展能力有限，最大100台
8. **数据规模**：相比较于MySQL来说可以支持很大规模的数据

## <font color=red>Hive安装与配置</font>
### Hive安装
1. [**官网地址**](http://hive.apache.org/)
2. [**文档查看**](https://cwiki.apache.org/confluence/display/Hive/GettingStarted)
3. [**下载地址**](http://archive.apache.org/dist/hive/)
4. **基本操作**
```SQL
show database # 展示数据库，显示有一个default数据库
use default # 使用default数据库
create table student(id int,name string) # 建表
show tables # 有student表
select * from student # 查询
insert into table student values(1，'banzhang'); # 插值
```

### 文件系统中导入Hive案例
```SQL
create table student(id int,name string) row format delimited fields terminated by '\t'; # 注意基于文件插入内容时，需要在建表的时候指定文件行分隔符，否则依据文件插入内容是会有问题

load data local inpath 'xxx' into table student; # 将xxx文件导入到student表中，如果在HDFS上，则去掉local，非本地加载。其实load本质就是hadoop里面的put
# hadoop fs -put stu1.txt xxx # 如果表中缺少数据，需要额外加入，可以使用hadoop进行put，select * 一样可以读进来
quit; # 退出Hive
```

### 安装MySQL
由于只能打开一个Hive（derby），因此需要安装MySQL，使用MySQL的Metastore，保证打开多个Hive
1. 将mysql中连接的jar包复制到hive/lib下
2. 修改配置，包括同其他设备的连接，包括打印表名等设置

### Hive交互命令
```sql
bin/hive -help # 查看命令，主要有两个参数，-e，-f，前者是命令行，后者是文件
bin/hive -e "select * from aa;" # 可以直接不进入Hive进行结果返回
bin/hive -f xx.hql > xx.hql.result.txt # 将xx.hql命令中的结果写到xx.hql.result.txt中
```

### Hive其他命令
```sql
bin/hive # 启动hive
dfs -ls /; # 查看hdfs里面文件内容
! ls ./ # 本地文件内容
```

<font color=red>注：由于Hadoop没怎么接触过，有一些配置没太搞懂，后续需要重新学习</font>

## <font color=red>Hive 数据类型</font>
### 基本数据类型
1. 常用的有INT，BIGINT，DOUBLE，STRING，分别对应于java中int，long，double，string，大小写没关系，其中string理论上可以存到2G的数据

### 集合数据类型
1. STRUCT，类似于c语言中的struct，可以通过"点"来访问，.first这种
2. MAP，是键值对元组集合
3. ARRAY，数组类型

**注：比较少用，一般会使用自定义函数进行拆分，形成扁平化的数据。Hive只能识别一行一行的数据，JSON之类的格式是无法识别的。**

*一个例子*
```bash 
# 文件中的一行内容：
# songsong,bingbing_lili,xiao # song:18_xiaoxiao song:19,hui long guan_beijing 

# 建表命令
# 注意不同内容间间隔符要一致，例如：无论array还是map还是struct中多个元素都由'_'分割，因为都是集合类型。如不一致，会有问题，所以需要提前处理文件，保证文件后面读入较为顺利。
create table test(
name string,
friends array<string>,
children map<string, int>
address struct<street:string, city:string>
)
row format delimited fields terminated by ','
collection items teminated by'_'
map keys terminated by ':'
lines terminated by '\n';

# 将文件内容载入表中
load data local inpath 'xxx' into table test; 

# 访问集合元素
select friends[1] from test; # 访问friends列第二个元素
select children['xiaosong'] from test; # 查看children列中xiaosong对应的value，与array的区别在于，array使用数字角标，map使用key值
select address.city from test # struct访问方式使用 "."+属性名
```
### 类型转化
1. 隐式类型转换
    a. TINYINT -> INT; INT -> BIGINT
    b. 整数类型、FLOAT、STRING -> DOUBLE，注意STRING必须要是数值类型的
    c. 整形除BIGINT -> FLOAT
    d. BOOLEAN不可转其他类型
2. 强转
    a. cast('1' as int)，得到int类型1
    b. cast('x' as int)，无法强转会失败，失败之后返回空值NULL
    
## <font color=red>DDL数据定义</font>
**Data Definition Language 数据定义语言，主要包括创建、修改、删除行为**
### 数据库的增删改查
#### 数据库的增
```bash
create database hive; # 创建数据库，其中hive也可以写成 库名.表名 这种形式，表示在库名下建某个表名
create database if not exists hive; # 创建数据库，需要判断是否已经有该数据库了
create database hive2 location '/hive2' # 在根目录下建一个hive2的数据库，之后建的表在/hive2文件夹中，文件名和库名可以不一样，不一样的时候默认是default
use hive; # 使用该数据库
create table test(id int) # 在hive库中建test表，其中有一个字段叫做id类型是int型
```
#### 数据库的删
```bash
drop database hive; # 将空数据库hive删除
drop database hive cascade; # 当hive库中存在表的时候，上条命令无法直接删除，需要加cascade进行强删
```
#### 数据库的改
```bash
alter database hive set dbproperties('createtime'='20200228') # 注意，所谓对数据库的改动，是对数据库以key-value对加一些属性，但无法改变相应的数据库名称、数据库位置
```
#### 数据库的查
1. 查数据库
```bash
show databases; # 打印存在数据库
show databases like 'hive*'; # 模糊查询，前缀为hive的数据库
```
2. 查数据库详情
```bash
desc database hive; # 显示数据库的信息，包括名称、位置、所有者之类的
desc database extended hive; # 显示数据库的信息，扩展信息，后续加入的信息可以查看
```

### 表的增删改查
表内容的查询命令较为复杂，因此后面章节再讲。

#### 创建普通表
1. 基本语法
```bash
CREATE [EXTERNAL] TABLE [IF NOT EXISTS] table_name 
[(col_name data_type [COMMENT col_comment], ...)] 
[COMMENT table_comment] 
[PARTITIONED BY (col_name data_type [COMMENT col_comment], ...)] 
[CLUSTERED BY (col_name, col_name, ...) 
[SORTED BY (col_name [ASC|DESC], ...)] INTO num_buckets BUCKETS] 
[ROW FORMAT row_format] 
[STORED AS file_format] 
[LOCATION hdfs_path]
# EXTERNAL表明创建的表为外部表与之相对的为管理表
# 其中COMMENT为添加注释信息
# PARTITIONED为分区，将一套数据分成多个文件夹
# CLUSTERED为分桶，将一套数据分成多个文件
# ROW FORMAT对文件格式进行限制
# LOCATION为库存储位置
```

2. 管理表（内部表）、外部表区别
内部表与外部表是通过建表时是否有EXTERNAL决定的，两者的区别：内部表认为原始数据是内部管理的，外部表则不是。翻译成人话，删除内部表的时候meta data与original data全部删除，外部表，只删除meta data。
```bash
drop database hive; # 删除表
show tables; # 此时无论外部表还是内部表，删除的表信息都没了，但是此时外部表源数据还在。当重新构建外部表的时候，使用select * from xxx是可以看见删表之前的数据的（原因：整体的流程是先从meta data中寻找源数据信息，再去源数据中调，源数据与元数据创建先后无关）。
```
```bash
show tables; # 展示表
show create table cc; # 展示表cc中详细信息
```
**注：临时表可以建成内部表，其他表一般建成外部表较为安全，防止删库跑路。**

3. 内部表外部表转换
```bash
desc formatted dept; # 查看dept表的详细信息，其中有表的类型信息，属于内部表还是外部表
alter table dept set tblproperties('EXTERNAL'='TRUE'); # 将manage table变成外部表，注意'EXTERNAL'='TRUE'是区分大小写的!!!如果小写，相当于加了一个属性，即external=true属性
```
#### 创建分区表
1. 分区表含义及作用
分区表实际就是HDFS系统上的一个独立文件夹，正常一个表是一个文件夹，但是分区表之下还有一个文件夹存储着不同的基于该表的分割表。分区表的作用是为了加速操作，例如：以时间进行分区，如果select某字段的时候局限在某个时间点，可以只在该分区下进行搜索，不必整张表搜索。通过where关键词进行筛选，称作谓词下推，执行的时候先from找表，再where过滤。
```bash
# 创建分区表
create table dept_partition(no int, name string, loc string
)
partitioned by (month string)
row format delimited fields terminated by '\t'

# 载入数据
load data local inpath 'xxx1' into table dept partition(month='202001')# 对该条数据加载，并指定分区为202001。实际产生了一个为month=202001的文件夹，里面存储了加载的数据。
load data local inpath 'xxx2' into table dept partition(month='202002') # 同上
select * from dept_partition; # 上述两个分区的数据都会显示，其中month会作为一个列存在。注意需要和sql中的index区分，index是存在表中的信息，HIVE是以文件夹进行存储的。
select * from dept_partition where month='202001' # 只显示分区信息为202001的数据，换句话说只扫描了month=202001文件夹
```

2. 增加分区表
```bash
alter table dept_partition add partition(month='2020-03') partition(month='2020-04') # 可以添加多个分区，实际上就是在dept_partition库所在文件夹中添加多个文件夹。mack repair table dept_partition 多文件恢复 partition(month='2020-04') # 可以添加多个分区，实际上就是在dept_partition库所在文件夹中添加多个文件夹。mack repair table dept_partition 多文件恢复c partition(month='2020-04') # 可以添加多个分区，实际上就是在dept_partition库所在文件夹中添加多个文件夹。mack repair table dept_partition 多文件恢复 partition(month='2020-04') # 可以添加多个分区，实际上就是在dept_partition库所在文件夹中添加多个文件夹。mack repair table dept_partition 多文件恢复c。
hadoop fs -put file xxx/month=2020-03 # 将文件放在分区文件夹之后，可以使用select访问 
```

3. 删除分区
```bash
alter table dept_partition drop partition(month='2020-03') # 删除分区
alter table dept_partition drop partition(month='2020-03'),partition(month='2020-04') # 删除多个分区需要逗号隔开，区别于添加时的空格
```

4. 查看分区
```bash
show partitions dept_partition; # 查看分区
```

5. 创建二级分区
```bash
# 其实就是多加一个字段，相当于多加一层文件夹
create table dept_partition(no int, name string,loc string
)
partitioned by (month string, day string)
row format delimited fields terminated by '\t';
# 过滤数据
select * from dept_partition where month='2020-07' and day='01' # 二级分区的过滤
```

#### 表的修改
1. 重命名
```bash
alter table old rename to new; # 将表名old修改为new
```
2. 增加删除分区
```bash
alter table dept_partition add partition(month='2020-03') # 增加分区
alter table dept_partition drop partition(month='2020-03') # 删除分区
```
3. 增加、修改、替换列名
```bash
# add 和 replace 可以多个
alter table test add columns (name string);
# change 只能改一个
alter table test change column name sex string; # 将name列改成sex，类型必须写上去。名称和类型可以与之前相同也可不同
# replace是对所有列名都修改
alter table test replace columns (a int); # 换句话说，无论之前有多少列，现在都改成一列a类型为int，原数据不变，元数据变化。如果只有一列，那么只读原数据的第一列，类型和原数据不匹配返回NULL。
desc test; # 查看表结构
```

#### 表的删除
```bash
drop table test; 
```

## <font color=red>DML数据操作</font>
**Data Manipulation Language 数据操作语言，主要包括select，update，insert，delete等**

### 数据的导入
1. load数据
```bash
load data [local] inpath '/opt/module/datas/student.txt' [overwrite] into table student [partition (partcol1=val1,…)]; # local是本地导入，如果不写，远端HDFS导入；overwrite是覆盖，直接into是追加；partition是追加分区，追加的内容是写在表中靠上的位置的。
```

2. insert数据
```bash 
# 创建表
create table student (id int, name string) partitioned by (month string) row format delimited fields terminated by '\t';
# 基本插入
insert into table student partition(month='202002') values(1, 'zhangsan') # 会走MR
# 查询插入
insert into table student
select id from test; # 将test的id插入到student表中
insert overwrite table student
select id from test; # 将test的id插入到student表中，注意会覆盖！与load区分，没有into
# 创建表基于select插入数据
create table test1
as select id from test; # 1.创建表；2.在表中加入select筛选出来的数据
# 创建表基于location插入数据
create table test2(id int, name string)
row format delimited fields terminated by '\t'
location 'xxx' # 其中xxx文文件路径
# import数据到指定表中
import table test partition(month='202002') from 'xxx' # 注意xxx一定要是export导出路径，随便一个文件是不行的，要使用load
```

### 数据的导出
1. insert 导出
```bash
# 查询结果格式化导出到本地/HDFS
insert overwrite local directory 'xxx'
row format delimited fields terminated by '\t'
select * from student# 结果导入本地,导入HDFS去掉local，不要忘记分隔符
```

2. HIVE SHELL命令导出
```bash
bin/hive -e 'select * from test' > xxx.txt # 将test表中数据导出到xxx.txt文件
bin/hive -f '...' > xxx.txt # 执行脚本进行文件内容的导出
```

3. export 导出到HDFS
```bash 
export table test to 'xxx' # 将test表导出到xxx路径，相当于copy，导出了元数据和原数据
import table test from 'xxx' # 配合之前export导出结果使用，表格式必须要是数据空的，或者test不存在
```

4. 清除数据
```bash
truncate table test; # 将原数据删除，保留元数据即表结构，因此只能操作内部表（管理表）
```

## <font color=red>数据表查询</font>
### 基本查询
#### 全表及特定列查询
```bash
# 查询全表
select * from test;

# 特定列查询
select id,name from test; 
```
**规则：**
**1. 大小写不敏感**
**2. 可以一行也可多行**
**3. 关键字不能缩写不能分行**
**4. 字句一般分行写**
**5. 分号表明语句结束**

#### 列别名
```bash
select ename as name, dbid id from test; # 查询结果的列名改成需要的名字，as可加可不加
```

#### 算数运算符
运算符：+ - * / % &  | ^ ~
分别是：加/减/乘/除，取余，按位与/或/异或/非
```bash
select id+1 from test; #加操作
```

#### 常用函数
```bash
select count(*) cnt from test; # 查看数量
select max(salary) maxsal from test; # 查看最大值
select min(salary) minsal from test; # 查看最小值
select sum(salary) sumsal from test; # 查看总和
select avg(salary) avgsal from test; # 查看平均值
select * from test limit 5; # 对查询结果只返回5条
```
**注：会触发MR任务**

### where语句
1. 基本where
```bash
select * from test where sal>1000; # 过滤
```

#### 比较运算符
```bash
a=b # 相等，ab中有一个为null都返回null；select null='a'会返回null
a<=>b # 大部分时和=一样，当两个都为null时，返回true，一个为null返回false
a!=b # 不等
a<b,a<=b,a>b,a>=b; # 同=符号，其中有一个为null返回null
a [not] between b and c; # 区间内，加not就是非区间内。eg:select 'b' between 'a' and 'c'，返回true。注意between是左右都闭的区间。
a is [not] null # 判断a是否为null
in #查看是不是在一个集合里面，eg:select 'a' in ('a','b')返回true
```

#### Like和RLike比较运算符
```bash
# _ % 分别匹配单个字符和多个字符
select 'a' like 'a_' # 返回false，因为_为占位符必须对应一个字符
select 'a' like 'a%' # 返回ture，因为是多个字符
select * from test where sal RLIKE '[2]'; # RLIKE是java中的正则表达式，这句话是说匹配sal字段中包含2的那些
```

#### 逻辑运算符
包含and or not
```bash
select * from test where sal>100 and sal<1000; # 使用and
select * from test where sal>1000 or id<300; # 使用and
select * from test where id not IN(20, 30); # 使用and
```

### 分组
#### group by函数
group by，一般和聚合函数使用
```bash
select t.id, avg(t.sal) avg_sal from test t group by t.id # 查询某个部门id的平均工资
select t.id, t.job, avg(t.sal) avg_sal from test t group by t.id,t.job # 查询某个部门id某种工作的的平均工资，可groupby多个
```

#### having函数
having和group by的区别：having与group by公用；where后面不能接分组函数，having可以；where针对表中列起作用，having针对查询结果中的列起作用，换句话说where是先筛选，having是对group by之后的结果筛选
```bash
select id, avg(sal) avg_sal from test group by id; # 查看不同部门id的平均薪水
select id, avg(sal) avg_sal from test group by id having avg_sal>2000; # 查看平均薪资大于2000的部门
```

### 连接
#### join两表连接
    a. 只支持等值连接，非等值连接不支持(!= > < >= <=等)
    b. 表名可以简化，简化的一个好处是提高效率，因为写明之后直接可以定位到相应的表上，多表连接的时候尤其有用
```bash 
select t1.id, t1.name from test1 t1 join test2 t2 on t1.id=t2.id; # 将两个表按照id进行join，内连接。
select t1.id, t1.name from test1 t1 left join test2 t2 on t1.id=t2.id; # 将两个表按照id进行join，左连接。
select t1.id, t1.name from test1 t1 right join test2 t2 on t1.id=t2.id; # 将两个表按照id进行join，右连接。
select t1.id, t1.name from test1 t1 full join test2 t2 on t1.id=t2.id; # 将两个表按照id进行join，全外连接。
select eid, dname from emp,dept; # 返回所有eid和dname的排列组合。当不给定连接条件的时候，数据表之间会进行笛卡尔积操作，即两表数据条目数相乘，当表中数据量比较大的时候相当危险。

# 不支持or连接，会报错
select e.no, e.name, d.no 
from emp e 
join dept d on e.no=d.no or e.name=d.name;
```

#### join多表连接
连接n个表至少需要n-1个连接条件
A join B on ... join C on ...
```bash
select e.name, d.deptno, l.locname
from emp e
join dept d on d.deptno=e.deptno
join location l on d.loc=l.loc;# 注意先启动1个MR对d、e表进行连接，再启动一个MR对d、l连接，原因是HIVE总是按照从左至右顺序执行的
```

### 排序
#### 全局排序
```bash
select * from emp order by sal asc; # 按工资排序，默认升序
select * from emp order by sal desc; # 按工资排序，降序排列
# 当出现order by的时候只有一个reducer
```

#### 按照别名排序
```bash
select ename, sal*2 twosal from emp order by twosal; # 薪水两倍排序
```

#### 多字段排序
```bash
select id, name from test order by id, sal; # 先按id排序，再按sal排序
```
 sort by排序
sort by只能对每个reducer进行排序，不是全局哦
```bash
set mapreduce.job.reduces=3; # 设置
select * from test sort by id desc; # 结果不太明朗
insert overwrite local directory 'xxx'
select * from test sort by id desc; # 写入文件，注意分3个文件，因为mapreduce之前设置了三个，每个文件中id都是降序的，但是id并不区分
```

#### 分区排序
结合sort by使用，用于sort by之前
```bash
set mapreduce.job.reduces=3; # 一定要分配多reduce
insert overwrite local directory 'xxx'
select * from emp distribute by deptno
sort by empno desc; # 基于deptno分区排序，并基于empno排序。写入文件夹中有三个文件，不同deptno分布在不同文件，文件内部按照empno降序
```

#### cluster by
```bash
# 下面两条命令是一样的，相当于是分成多个分区，每个分区内部都按照deptno排序
# 注意：cluster by只能按升序排序，不能降序
select * from emp cluster by deptno; 
select * from emp distribute by deptno sort by deptno; 
```
**注：当reducer是n个，但是distribute by的列名种类数大于n个时，会随机分配到n个文件中**

### 分桶及抽样查询
分区和分桶，分区是放在不同的文件夹中，分桶是在同一个文件夹中多个文件，distribute by类似于分桶
```bash
# 分区字段是新的，分桶字段是表中有的字段
# 分桶建表
create table stu buck(id int, name string)
clustered by(id) into 4 buckets
row format delimited fields terminated by '\t' # 注意load和insert都没办法建立分桶效果，因为load是put一个文件，不可能分多份put

create table stu(id int, name string)
row format delimited fields terminated by '\t';
load data local inpath '/opt/module/datas/student.txt' into table stu; # 文件内容放到一个普通表
set hive.enforce.bucketing=true; # 设置开启分桶
set mapreduce.job.reduces=-1;
insert into table stu_buck
select id, name from stu; # stu是之前的一个普通表，从文件中读取的
```
**注：一个小窍门，对于查询的都不带ed，对于建表之类的需要带ed。比如：建分桶表clustered，数据排序查询使用cluster，再如partitioned by**
```bash
# tablesample(bucket x out of y on id)
# 注意x<y，其含义是从x开始抽（从1开始），一共抽z/y个，z为设定的表的桶数
select * from stu_buck tablesample(bucket 1 out of 4 on id); # 分桶抽样
```

### 其他常用查询函数
#### 空字段赋值
```bash
select nvl(sal, -1) from emp; # 查询emp表中的sal这列，如果值为NULL，则赋值为-1
select nvl(sal, id) from emp; # 查询emp表中的sal这列，如果值为NULL，可以使用id类替代
```

#### 时间类
```bash
# 格式化时间
select date_format('1987-5-23', 'yyyy-MM') # 返回对应年月
select date_format('1987-5-23', 'yyyy-MM-dd HH:mm:ss') # 返回对应年月日，时分秒
select date_format('1987-5-23', 'yyyy-MM')<'2020-01' # 年月可以比较

# 时间和天数相加
select date_add('2020-02-29', 5) # 返回2020-03-05，相当于对当前时间进行后推
select date_add('2020-02-29', -5) 

# 时间和天数相减
select date_sub('2020-02-29', 5) # 和上面记一个就行

# 两时间相减
select datediff('2020-02-29', '2020-02-27')

select regexp_replace('2020/02/15','/','-') # 字符替换，将/进行替换
```
**注：date_format, date_add, date_sub, datediff都只认以-分割的时间格式，需要regexp_replace替换**

#### CASE WHEN
```bash 
# 查看emp表中按deptid分组后，各组男女数目
select deptid,
sum(case sex when '男' then 1 else 0 end) male_cnt,
sum(case sex when '女' then 1 else 0 end) female_cnt
from emp group by deptid;

# 由于只有两种条件选择，可以使用if-then形式
select deptid, 
sum(if(sex='男',1,0)) male_cnt,
sum(if(sex='女',1,0)) female_cnt
from emp group by deptid;
```

#### 列转行
```bash
select concat('hello', '-', 'world') # 拼接字符串
select concat(id, '-', name) from test; # 对test的两列id和name进行拼接

select concat_ws('-', 'hello', 'world'); # 对字符串进行拼接
select concat_ws('-', id, name, address) from dept; # 对dept中id和name两列进行合并，注意分隔符写在最开头

select collect_set(id) from dept; # 对dept表中id这列去重，注意也有collect_list函数，即数组

select concat_ws('_',collect_set(id)) from dept; # 对dept表中id这列去重，并使用concat函数进行聚合。数字可以使用cast强转

# 一个实例：
# 北京 a 嘻嘻     北京,a 嘻嘻|呵呵
# 上海 b 哈哈     上海,b 哈哈|桀桀
# 北京 a 呵呵 =>  广州,c 咯咯
# 广州 c 咯咯    
# 上海 b 桀桀    
# 拆成两部分，先是前两列进行合并，再进行groupby
# 第一步
select 
    concat_ws(',', address, name) address_name, laugh 
from 
    test; # t1
# 第二步
select 
     address_name, concat_ws('|', collect_set(laugh))
from 
    t1
group by address_name
;
# 完整步骤
select 
     address_name, concat_ws('|', collect_set(laugh))
from 
    (select 
    concat_ws(',', address, name) address_name, laugh 
from 
    test)t1
group by address_name; # 将第一步第二步合并的code
```

#### 行转列(explode)
```bash
# 一个例子:
# movie	category
# 《疑犯追踪》	悬疑,动作,科幻,剧情
# 《Lie to me》	悬疑,警匪,动作,心理,剧情
# 《战狼2》	战争,动作,灾难
# =>=>=>=>=>=> 转换成
# 《疑犯追踪》      悬疑
# 《疑犯追踪》      动作
# 《疑犯追踪》      科幻
# 《疑犯追踪》      剧情
# 《Lie to me》   悬疑
# 《Lie to me》   警匪
# 《Lie to me》   动作
# 《Lie to me》   心理
# 《Lie to me》   剧情
# 《战狼2》        战争
# 《战狼2》        动作
# 《战狼2》        灾难
select explode(category) from movieinfo; # 会将category这一列按照顺序整体展开
select 
    movie movie_cat 
from 
    movieinfo lateral view explode(category) tmp as movie_cat; # 注意，需要lateral view和explode连用，最后写上列的别名，tmp是表的别名
```
#### 窗口函数
```bash
select name,count(*)
from business
where substring(date, 1, 7)='2017-04'
group by name; # 显示的是name及其对应出现的次数

select name,count(*) over()
from business
where substring(date, 1, 7)='2017-04'
group by name; # 显示的是name及当前不重复的name的总数量。换句话说相当于对去掉count(*) over()之后的结果进行count(*)，因为事先groupby，所以就是name的类别数

select date,cost,sum(cost) over(order by date)
from business; # 对date排序之后，对每一条进行开窗，即求所有日期的累和操作。如果不加order by那么对查询结果所有cost求和，即所有条目的值都相等。

select name, date, cost, sum(cost) over(distribute by name) 
from business; # 展示每条购物明细，并对每位顾客的消费总额进行展示。不能用group by，需要替代为distribute by。

select name, date, cost, sum(cost) over(distribute by name sort by date) 
from business; # 展示每条购物明细，并对每位顾客不同天的消费记录按时间顺序进行累加。distribute by与sort by连用。

select name,orderdate,cost, 
lag(orderdate,1,'1900-01-01') over(partition by name order by orderdate ) as time1
from business; # 查询每个顾客的购物明细及上一次购买记录的时间。lag函数统计窗口内往上第n行值，第一个参数为字段名，第二个参数为n，第三个参数为缺失值填充，如不指定，默认为NULL。注意partition by name order by orderdate可以替换为distribute by ... sort by，同时末尾可以加desc或者asc。

select name,orderdate,cost,  lag(orderdate,1,'9999-99-99') lead(partition by name order by orderdate ) as time1
from business; # lead是统计窗口内向下第n行值。

select name, orderdate, cost, ntiles(5) over() ntile5
from business; # 该命令的作用是对所有数据行进行5等分，并且对每一行进行编号，从1开始，如果行数不足以整除，最后的编号可能会较之前的编号少。
# 该命令的使用场景为：当求某个数据前n%的数据时，可以在ntiles对应的括号中写入1/n%对应的数字，例如：求前20%的数据，可以使用ntile(5)，即分5份取1份，需使用子查询。

# 注意：lag，lead，ntile函数后面必须跟over字段

# 注意：over内部可以使用关键字：current row，n preceding，n following，unbounded preceding，unbounded following，分别表示当前行，前n行，后n行，从头开始，到尾部。结合rows between ... and ...使用，是左闭右闭的区间，注意and前面一定大于其后面的。
# 例子：
select
name, orderdate, cost, sum(cost) over(rows between 2 preceding and current row)
from business; # 表示查看原始数据每条数据及其前两条数据之和，partition by写在括号内rows前面

```
>**总结来说，over是对每个窗口(默认每条)进行相应聚合函数使用的命令，我感觉有点类似于pandas里面的groupby+apply+自定义函数，都是在指定窗口内进行操作，并返回“1”个结果。over的行数与除over所在行之外的命令所执行结果对应的行数相等，如果想去重，建议使用group by。**

>**HIVE命令写的顺序：select，from，join，where，group by，order by，having，limit**

>**HIVE命令执行的顺序：from，on，join，where，group by，having，select，order by，limit。命令中存在子查询的先做子查询，提示我们做表的join的时候如果有过滤条件可以将过滤条件写进子查询，之后再join。**

#### 排名函数(窗口函数)
> 排名函数也属于窗口函数
> 总共有三种，分别为**rank()**, **dense_rank()**, **row_number()**,三者的区别在于对排序相同时的处理方式。
> 举例：当4人的数学成绩为99，99，98，97时，要求对上述成绩进行排名。
> 
| 函数  | 结果     | 解释  |
| ------------- |:-------------:| -----|
| rank() | 1,1,3,4 | 排序相同时会重复，总数不变  |
| dense_rank() | 1,1,2,3      |  排序相同时会重复，但总数会减少 |
| row_number() | 1,2,3,4      |  会根据顺序计算 |
```bash
select 
    name ,subject, score,
    rank() over(partition by subject order by score desc) rank1,
    dense_rank() over(partition by subject order by score desc) rank2,
    row_number() over(partition by subject order by score desc) rank3
from 
    score; # 分别使用三种函数
# 使用的场景：某个店铺访问前3的用户信息，通过rank得到用户排名，之后使用where对rank小于4的过滤即可。limit不好用于分区，一般用于整体。
```

## <font color=red>函数</font>
### 系统内置函数
```bash
show functions; # 查看系统自带函数
desc function upper; # 显示自带函数的用法
desc function extended upper; # 详细的显示自带函数的用法
```
### 自定义函数
> 自定义函数主要有三种类型：
> 1. **UDF**(user-defined-function): 一进一出，如datediff。
> 2. **UDAF**(user-defined-aggregation-function):多进一出，如max。
> 3. **UDTF**(user-defined-table-generating-function):一进多出，如explode。

> 编程步骤：
> 1. 继承org.apache.hadoop.hive.ql.UDF
> 2. 需要实现evaluate函数，evaluate函数支持重载
> 3. 在HIVE的命令行窗口创建函数
    > a. 添加jar包：
    > **add jar linux_jar_path**
    > b. 创建function:
    > **create [temporary] function [dbname.]function_name AS class_name # 使用temporary表示退出之后该功能删除，dbname表示对应库名。dbname指定了自定义函数使用的库，如果换了数据库(use xxx)会显示函数未定义。**
> 4. 在HIVE的命令行窗口删除函数
> **Drop [temporary] function [if exists] [dbname.]function name**
> **注意UDF必须要有返回类型，可以是NULL，但不能是void**
> 5. 使用：
> **select myfun(id) from test; # 对test表中id这列使用自定义函数**


```bash

```


## <font color=red>其他</font>
1. distinct与双group
```bash
双group去重可能会稍微快一点
# distinct
select
    shop,
    count(distinct id) user_cnt
from
    visit
group by 
    shop;

# 双group
select 
    shop,
    count(*) user_cnt
form
    (
select 
    shop, id
from
    visit
group by
    shop,id)t1
group by 
    shop;
```
2. 表中的列可以使用“.”来取用，例如：test表中的id列，test.id
3. select ... from t1,t2,t3 from可以取多个表
4. floor()向下取整
5. 网上有对sql进行格式优化的工具，可读性高
6. 能先过滤数据集，就先过滤，方便效率的提高
7. 连续n天问题思路：利用row_number + over对日期给定排序后的rank值，之后，使用date_sub对日期和rank值相减，再使用sum + over对上步相同日期差值组计算行量，最后过滤数量>=n对应的数据行即可。（详见b站-尚硅谷HIVE-蚂蚁森林解法二）
8. sum(消费) + over(partition by 用户+ order by 时间+ rows between unbounded preceding and current row)可以用于对用户按时间顺序累积的消费进行统计。
9. word-count思路：split + explode + group by + count

##

<hr />
