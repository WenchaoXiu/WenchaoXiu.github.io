---
title: scikit-learn学习笔记
tags: []
date: 2019-04-11 09:44:48
permalink:
categories: Machine learning
description:
image:
---
<p class="description">机器学习算法的实践笔记</p>

<img src="./images/sklearn.jpg" alt="" style="width:100%" />

<!-- more -->
# 笔记简介
主要整理一下近期学习的机器学习算法具体的实现过程,主要使用的package有scikit-learn,xgboost

# sklearn各功能实现
## 数据预处理
1.对类别数据进行离散化
```Python
# 因为构件数学模型的时候类别特征可以取多个值，不同的值实际上代表的是不同特征属性，因此需要对类别特征进行离散化处理
# 1.利用pandas的get_dummies进行onehot编码
newDF = pd.DataFrame()
for col in DF.columns:
	tmp = pd.get_dummies(DF[col])
	tmp = pd.rename(columns = lambda x: col+'_'+str(x))
	newDF = pd.concat([newDF, tmp], axis=1)
x = newDF.values

# 2.利用sklearn的label_binarize进行onehot编码
# 除了上述的pd.get_dummies之外的另一种等价方法
from sklearn.preprocessing import label_binarize
DF_col = label_binarize(DF[col], classes = np.arange(len(pd.unique(DF[col]))))
# label_binarize返回的和pd.get_dummies返回的对象类似，列数与类别数相同，且只包含01值
# 区别在于，label_binarize返回的是ndarray数据，pd.get_dummies返回的是DataFrame数据
# 因此，np.array_equal(DF_col, pd.get_dummies(DF[col]).values)返回的是True

# 3.sklearn的OneHotEncoder
# 这个功能略感鸡肋(也可能是打开方式不对)给个例子，就不赘述了
from sklearn.preprocessing import OneHotEncoder
x = [[1,2,3,4],
     [2,1,3,3],
     [3,4,1,2],
     [1,1,1,1],]
ohe = OneHotEncoder(sparse=False)
print ohe.fit_transform(x)

输出：
[[1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0.]
 [0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 0. 0.]
 [1. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0.]]
```

2.对响应变量进行转换，将类别数据转化为不同int型数字
```Python
y = DF.accept
y_int = pd.Categorical(y).codes #得到从0~n-1的n类数字，分别对应不同的类别
# 与上述pd.get_dummies的区别在于，get_dummies返回的是DataFrame数据，且只有01两个值
# 但是上述方法返回的是ndarray数据，返回的值不止01，具体的根据类别数量进行确定

from sklearn.preprocessing import LabelEncoder
y_int = LabelEncoder().fit_transform(y_train)
# 实现了跟上面相同的功能，输入是不用类别的label，返回的是0~n-1的数字
```

3.训练集测试集分割
```Python
# 这样做的目的是方便测试算法泛化能力
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
# PS:其中x, y是ndarray数据类型，如果是DF数据的话需要使用DF.values进行转化
```




## 特征工程

1.PCA数据降维
```Python
from sklearn.decomposition import PCA
pca_model = PCA(n_components=2, whiten=True)
x_pca = pca_model.fit_transform(x)
print 'top2 方差', pca_model.explained_variance_
print 'top2 方差所占比例', pca_model.explained_variance_ratio_
# 其中的n_component是选择将数据降维后，选取其中的n个特征（特征顺序按照特征值从高到低排序）
# pca_model.explained_variance_是对应的前两个特征的方差
# pca_model.explained_variance_ratio_是前两个特征的方差占总方差的比例
import seaborn as sns
sns.lmplot(x = 'pc1', y='pc2', data = pc_plot_data, hue=type, fit_reg=False)
ax = plt.gca()
ax.set_title("Iris PCA 2 compotent", fontsize=20)
# 上述代码用来绘图
```
[使用教程](https://www.cnblogs.com/pinard/p/6243025.html)

2.特征筛选
```Python
# 利用sklearn的SelectKBest进行特征选择
from sklearn.feature_selection import SelectKBest, chi2
chi2_model = SelectKBest(chi2, k=2)
chi2_model.fit(x, y)
selected_col = chi2_model.get_support(indices=True)
# 之所以x和y都需要加进去是因为需要对每个特征和响应变量进行chi2检验，按照显著性排序，找到前两个特征（原特征）
# indices是为了返回对应的column数
```
[官方文档](http://sklearn.apachecn.org/cn/0.19.0/modules/feature_selection.html) & [使用教程](http://bluewhale.cc/2016-11-25/use-scikit-learn-for-feature-selection.html) & [特征提取](http://d0evi1.com/sklearn/feature_selection/)

3.原始特征变化
```Python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias=True, interaction_only=False)
x_poly = poly.fit_transform(x)
# 对数据特征进行变换，degree是特征的最高维数，include_bias加不加一列1，interaction_only是否只看有交互的特征
```






## 构建有监督模型
1.逻辑回归模型用于分类（Logistic Regression）
```Python
from sklean.linear_model import LogisticRegressionCV, Lasso
model = LinearRegeressionCV(Cs = np.logspace(-3,4,8), cv = 5, n_job = -1)
model.fit(x_train, y_train)
y_train_hat = model.predict(x_train) # 对训练集进行预测
y_test_hat = model.predict(x_test) # 对测试集进行预测
print model.C_ # 获得各类别最佳超参数
# LinearRegeressionCV是一种自动确定正则化超参数的函数，其中Cs是一系列的正则化参数值，一般需要取不同数量级的数字
# cv是进行cross validation对超参数最优解进行确定，n_job=-1是用上全部线程，这里默认的正则是L2 norm
# PS:注意在进行多分类的时候选择的方法是one vers rest也就是一对多进行多分类

# 注意除了自带的cross-validation方法，还可以用更加普遍的方法GridSearchCV
model = Lasso()
GS_Lasso = GridSearchCV(model, param_grid={'alpha':np.logspace(-3,4,8)}, cv=5)
GS_Lasso.fit(x_train, y_train)
print 'Lasso model最优参数: ', GS_Lasso.best_params_
```

2.决策树模型（DesitionTree）
```Python
# 对决策树的不同深度进行探究，求取对应的准确率
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
acc = []
for i in range(1,15):
    model.set_params(max_depth=i)
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, y_test_hat))
plt.plot(np.arange(1,15), acc, 'ro')
plt.ylabel('accuracy')
plt.xlabel('depth')

#决策树进行回归
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
dt = reg.fit(x, y)
```

3.随机森林模型（RandomForest）
```Python
from sklearn.ensemble import RandomForestRegressor
RF_model = RandomForestRegressor()
RF_model.fit(x_train, y_train)
y_train_hat = RF_model.predict(y_train)
# 对于多值输出问题来说一般采用多值输出的方法进行模型构建，主要从两个角度：a.每个叶节点存储多个值 b.通过计算多个值的平均减少量作为split标准
```
[参考](http://sklearn.apachecn.org/cn/latest/modules/tree.html) & [随机森林调参](https://www.cnblogs.com/pinard/p/6160412.html)

4.bagging模型
```Python
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
DT_model = DecisionTreeRegressor(max_depth=9)
ridge_BG_model = BaggingRegressor(Pipeline([('poly', PolynomialFeatures(degree=6)), 
                                            ('ridge', linear_model.RidgeCV(alphas=np.logspace(-3,4,8), cv=5, fit_intercept=False))])
                                  , n_estimators=n_estimators, max_samples=max_samples)
DT_BG_model = BaggingRegressor(DT_model, n_estimators=n_estimators, max_samples=max_samples)
# bagging是一种构建多个基分类器，对数据进行预测，并平均多个分类器的预测值的一种方法
# 输入需要有基分类器，基分类器的个数，以及对于样本选取的比例
```

5.Adaboost模型
[原理](http://www.cnblogs.com/pinard/p/6133937.html) & [模型参数](https://www.cnblogs.com/pinard/p/6136914.html)
```Python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
base = DecisionTreeClassifier(n_estimators=100, max_depth = 3, min_sample_split=4, random_state=1, oob_score=True) 
# 超过4个样本就进行样本的分割构建叶节点, 基分类器个数100，最大树深3，oob_score考虑带外分数
AB_model = AdaBoostClassifier(base_estimator = base, learning_rate = 0.1)
AB_model.fit(x_train, y_train)
y_train_hat = AB_model.predict(x_train)
print '训练集准确度: %.5.f'%metrics.accuracy_score(y_train, y_train_hat)
# Adaboost是另一种集成模型，利用决策树模型作为基分类器，与上面的bagging的区别在于，bagging是可以并行的，各个模型的确定可以随机产生，最后平均化即可
# 但是对于Adaboost来说各个基分类器是串行产生的，每次新产生的分类器都与前面所有的分类器相关，具体是新的分类器是通过前面所有模型的残差和确定的
# 相当于不断强化基分类器与真实值的差别，因此Adaboost的基分类器需要更加泛化相比较Bagging的及模型
# 举个例子假如基分类器都是决策树，Bagging的深度更深，Adaboost的深度更浅
# 这里learning_rate实际上是fk(x)=fk−1(x)+ναkGk(x)中的ν(0<ν<1)，其中αk是根据分类器的错误率进行确定的模型权重
# Gk(x)是根据样本权重重新确定的基分类器，fk(x)是膜前获得的强分类器是k个基分类器的结合，理论上ν越小的话迭代下次数越多
```

6.GBDT模型
```Python
#GBDT 每个基分类器要相对弱一点因为是提升树
from sklearn.ensemble import GradientBoostingClassifier
gbdt_model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
gbdt_model.fit(x_train, y_train)
gbdt_y_train_hat = gbdt_model.predict(x_train)
gbdt_y_test_hat = gbdt_model.predict(x_test)
print '训练集准确率: %.4f'%metrics.accuracy_score(y_train, gbdt_y_train_hat)
```

7.xgboost模型
```Python
# xgboost也是一种快速效果好的集成模型
import xgboost as xgb
train_data = xgb.DMatrix(x_train, y_train)
test_data = xgb.DMatrix(x_test, y_test)
watch_list = [(test_data, 'eval'), (train_data, 'train')]
params = {'eta':0.1, 'max_depth':6, 'objective':'multi:softmax', 'num_class':3}
xgb_model = xgb.train(params, train_data, num_boost_round=50, evals=watch_list, evals_result={'eval_metric':'logloss'}) #设置对应评估指标
y_test_hat_xgb = xgb_model.predict(test_data) #######test_data类型，xgb_model跟sklearn不同
print 'xgb model: %.3f'%(sum(y_test_hat_xgb==y_test)*1.0/len(y_test))

# 利用XGBOOST的sklearn接口进行模型构建，这个方便点使用习惯和sklearn其他模型类似
xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=2, objective='multi:softmax')
xgb_model.fit(x_train, y_train)
xgb_y_train_hat = xgb_model.predict(x_train)
xgb_y_test_hat = xgb_model.predict(x_test)
print '训练集准确率: %.4f'%metrics.accuracy_score(y_train, xgb_y_train_hat)
print '验证集准确率: %.4f'%metrics.accuracy_score(y_test, xgb_y_test_hat)
```
[sklearn API](https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=fit#module-xgboost.sklearn) & [参数解释](https://blog.csdn.net/sb19931201/article/details/52557382) & [github使用教程](https://github.com/dmlc/xgboost/tree/master/demo#tutorials) & [GDBT和XGBOOSt区别](https://www.zhihu.com/question/41354392/answer/98658997) & [Tatanic实例](https://zhuanlan.zhihu.com/p/28663369) & [**知乎教程**](https://zhuanlan.zhihu.com/p/31182879) & [**Ensemble模型介绍**](https://zhuanlan.zhihu.com/p/26683576)

8.SVM模型[参考](https://www.cnblogs.com/pinard/p/6117515.html)
```Python
from sklearn import svm
model = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr', class_weight={-1:1, 1:weight[0]})
model.fit(x_train, y_train)
y_train_hat = model.predit(x_train)
print 'decision_function:\n', clf.decision_function(x_train)
print '\npredict:\n', clf.predict(x_train)
print '支撑向量的数目：', clf.n_support_
print '支撑向量的系数：', clf.dual_coef_
print '支撑向量: ', clf.support_ # 这个是对应的支持向量的索引值
print '模型精度: ', model.score(x_train, y_train)
# C是惩罚系数，当C很大的时候为了保证损失函数最小对应的惩罚因子需要很小，使得模型很严格，决策边界之间的距离很小
# class_weight是为了方便对不平衡数据的操作，如果类别数据数量级差的太多的话会使模型准确度下降
# kernel可以选择不同的核函数，decision_function_shape是进行one vers rest，可以使用ovo,这样的话一对一
# decision_function计算每个样本到每一类的距离，选取最大的那个作为预测类别（因为在决策边界两侧），decision_function的ovr和ovo列数有差别
# predict就是正常的返回类别对应的数字
# svm的score返回的是精度类似于metrics.accuracy_score(x_train, y_train)

weight = [2,30,2,30]
clfs = [svm.SVC(C=1, kernel='linear', class_weight={-1:1, 1:weight[0]}),
       svm.SVC(C=1, kernel='linear', class_weight={-1:1, 1:weight[1]}),
       svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={-1:1, 1:weight[2]}),
       svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={-1:1, 1:weight[3]})]
# 对于不平衡数据进行权重的分配，保证结果的准确性

# 利用SVM进行回归分析
from sklearn import svm
print('SVR - RBF')
svr_rbf = svm.SVR(kernel='rbf', gamma=0.2, C=100)
svr_rbf.fit(x, y)
print('SVR - Linear')
svr_linear = svm.SVR(kernel='linear', C=100)
svr_linear.fit(x, y)
print('SVR - Polynomial')
svr_poly = svm.SVR(kernel='poly', degree=3, C=100)
svr_poly.fit(x, y)
print('Fit OK.')
# 注意这里kernel可以选择linear/rbf分别对应线性核以及高斯核，其中线性核和高斯核分别有两个比较重要的参数
# C和gamma，其中C上面已经解释过了，gamma是一个跟核函数相关的参数，gamma是∂2的倒数，表征高斯核的方差大小，
# 所以∂2的倒数表征的是高斯核的精度，gamma值越大也就数据在训练集上准确度越高，增大过拟合的风险
# PS:kernel还可以选择poly多项式核，配合的参数是degree
```




## 构建无监督模型

1.Kmeans聚类[参考](https://www.cnblogs.com/pinard/p/6169370.html)
```Python
from sklearn import datasets as ds
x, y = ds.make_blobs(400, n_features=2, centers=4, cluster_std=(1, 2.5, 0.5, 2), random_state=2)
# 这个函数的功能是根据高斯分布进行构建随机数据，用于后续的聚类分析
# 其中400是产生的样本的个数，n_features是产生的数据的维度，centers是对应数据的类别，cluster_std是定义不同类别的方差
# 这里centers可以是一组array数据，作为中心

from sklearn.cluster import KMeans
model = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300)
model.fit(x, y)
y_hat = model.predict(x)
model.cluster_centers_ # 输出对应的聚类中心
# KMeans算法主要有这几个主要参数，n_cluster是聚类类别，init是中心点初始化方法，因为KMeans初值敏感所以需要迭代多次，
# n_init就是控制这个的参数，当数据是非凸集的时候max_iter的设置防止算法不收敛
```

2.AP聚类算法
```Python
from sklearn.cluster import AffinityPropagation
model = AffinityPropagation(affinity='euclidean')
model.fit(x, y)
model.cluster_centers_indices_ # 返回的是中心点的索引
y_hat = model.labels_ #返回的是预测的数据类别
```

3.DBSCAN算法([参考1](https://www.cnblogs.com/pinard/p/6208966.html),[参考2](http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py))
```Python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=m, min_samples=n) # eps对应算法的半径, min_samples对应算法在ε半径之内包含的最小点数，用来确定核心对象
model.fit(x)
y_hat = model.labels_ # 预测类别值
y_coresample = model.core_sample_indices_ # 预测类别中心点
n_clusters_ = len(set(y_hat)) - (1 if -1 in y_hat else 0) # 注意算法会把噪音点归为-1
# DBSCAN对数据本身没有分布要求，不像Kmeans假设数据是高斯分布的，DBSCAN产生的聚类形状可以不为类圆形
# 固定ε，min_samples越大越严格；固定min_samples，ε越小越严格
# DBSCAN通过计算每个数据点的高局部密度点距离和本身的密度确定每个点的类别，高局部密度点距离大密度大的点为聚类中心
# 高局部密度点距离大密度小的点为噪声，高局部密度点距离小密度大的点为普通点，高局部密度点距离小密度大的点不好判断
# ps:密度就是每个点ε邻域内的点数，高局部密度点距离就是比该点密度大的点中与之距离最小的距离
```

4.MeanShift聚类
```Python
from sklearn.cluster import MeanShift
model = MeanShift(bin_seeding=True, bandwidth=band_width)
# 只需要给出圆的半径就好，圆的中心根据圆圈内包含的点的中心不断更新

```








## 评估模型
1.模型准确率
```Python
from sklearn import metrics
print '训练集准确率: %.5f'%metrics.accuracy_score(y_train, y_train_hat) # 先放真实值再放预测值，类型是ndarray
print '测试集准确率: %.5f'%metrics.accuracy_score(y_test, y_test_hat) # 先放真实值再放预测值，类型是ndarray
# 这里y_train和y_train_hat都是由0、1组成的
```

2.模型的ROC、AUC[micro/macro](https://blog.csdn.net/YE1215172385/article/details/79443552)
```Python
from sklearn import metrics
micro_AUC = metrics.roc_auc_score(y_true, y_probability, average='micro')
macro_AUC = metrics.roc_auc_score(y_true, y_probability, average='macro')
fpr, tpr, threshold = metrics.roc_curve(y_true.ravel(), y_probability.ravel())
plt.figure(figsize=(8,6), dpi=80, facecolor='w')
plt(fpr, tpr, 'r-', lw=2, label='AUC: %.5f'%micro_AUC)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xticks(np.arange(-0.1, 1.1, 0.1))
plt.yticks(np.arange(-0.1, 1.1, 0.1))
plt.title('ROC curve', fontsize=18)
plt.grid(b=True, ls=':')
plt.legend(loc = 'lower right')

# 对上述评估标准进行说明，首先y_true和y_probability是一个n*m的ndarray对象，n是样本数，m是响应变量的类别数
# 其中y_true是对应多类别的真实值，每一行只有一个1其余为0，y_probability每一行是不同类别的预测概率
# 在计算AUC得分的时候，如果是多分类问题，有两种计算方式，第一种对每一列绘制ROC计算AUC，即对每一类别分别计算
# 另一种是将n*m的矩阵按行首尾连接，形成一个n*m长的array，对其绘制ROC同时计算AUC，这两种分别是macro和micro
# metrics.roc_curve是用来获得在不同theshold下得到的FPR和TPR的对应值，其中.ravel()方法用来展开数据，相当于按照micro的方法绘制
```

3.模型的MSE(Mean square error)[参考](http://cwiki.apachecn.org/pages/viewpage.action?pageId=10814010)
```Python
# 计算模型的均方误差
from sklearn import metrics
metrics.mean_squared_error(y_True, y_hat)
```

4.模型的其他评估(Mean square error) [参考](https://blog.argcv.com/articles/1036.c)
```Python
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
fbeta = fbeta_score(y_true, y_hat, beta=beta)
precision = precision_score(y_true, y_hat)
recall = recall_score(y_true, y_hat)
print('f1 score: \t', f1_score(y_true, y_hat))
fbeta = fbeta_score(y_true, y_hat, beta=beta)
# precision score = tp/(tp+fp)和准确度的区别在于只关注正类，准确率是所有类别都关注
# recal score = tp/(tp+fn)
# F1 = 2*(precision*recall)/(precision+recall) F1相当于P和R调和均值，越大模型效果越好
# fbeta_score是P和R的调和均值，beta<1 precision的权重更大，beta>1 recall的权重更大
```

5.聚类方法的评估([方法解释](https://blog.csdn.net/Mr_tyting/article/details/76719062)[参考1](https://blog.csdn.net/sinat_26917383/article/details/70577710), [参考2](http://sklearn.apachecn.org/cn/0.19.0/modules/clustering.html))
```Python
print('Homogeneity：', homogeneity_score(y, y_pred))
print('completeness：', completeness_score(y, y_pred))
print('V measure：', v_measure_score(y, y_pred))
print('AMI：', adjusted_mutual_info_score(y, y_pred))
print('ARI：', adjusted_rand_score(y, y_pred))
print('Silhouette：', silhouette_score(x, y_pred), '\n') # 轮廓系数，1-类内距离/最小类外距离，1的时候最优，真实值也不一定是1..
# 聚类算法在真实的应用中一般难以获取真实的数据标签，除非手动区分，所以上述评估标准比较鸡肋，不深入解释
```

## 其他
1.欧氏距离计算
```Python
from sklearn.metrics import euclidean_distances
m = euclidean_distances(data, squared=True)
# 返回一个n*n的矩阵计算的是亮亮之间的距离
```



## Pipeline构建
```Python
# pipeline的使用是为了方便对数据处理的流程化，举个简单的example:特征变换+逻辑回归
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
model = Pipeline([
	('poly', PolynomialFeatures(degree = 2, include_bias = True)),
	('LR', LogisticRegressionCV(Cs = np.logspace(-3, 4, 8), cv = 5, fit_intercept = False))
])
model.fit(x_train, y_train)
print '最优参数:  ', model.get_params('LR')['LR'].C_
# model包括两步，第一步对特征进行转换，第二步对转换好的特征进行逻辑回归模型构建
```
[Pipeline原理解析](https://blog.csdn.net/lanchunhui/article/details/50521648)



## 参数验证
```Python
from sklearn import svm
from sklearn.model_selection import best_estimator_, RandomizedSearchCV    # 0.17 grid_search
model = svm.SVR(kernel='rbf')
c_can = np.logspace(-2, 2, 10)
gamma_can = np.logspace(-2, 2, 10)
svr = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
svr.fit(x, y)
print('验证参数：\n', svr.best_params_)
# cross validation交叉验证对应的参数
# best_params_对应model的参数，对于内部的参数通过best_params_之后再提取，比如svr中的support_
```





## 分类数据进行绘图
```Python
# 1.对于多分类数据进行划分区域，对坐标轴中的各个点进行预测，并对预测的点给与相应的颜色，例如对x1，x2进行绘图
x1_min, x1_max = x1.min(), x1.max()
x2_min, x2_max = x2.min(), x2.max()
x1_array = np.arange(x1_min, x1_max, 0.02)
x2_array = np.arange(x2_min, x2_max, 0.02)
x1_tmp, x2_tmp = np.meshgrid(x1_array, x2_array)
x_plot = np.stack((x1_tmp.flat, x2_tmp.flat), axis=1) #x_plot是一个列为2的数据，第一列是x轴值，第二列是y轴值
y_meshgrid_hat = model.predict(x_plot)
plt.scatter(x_plot[:,0], x_plot[:,1], c=y_meshgrid_hat, cmap = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF']))
plt.scatter(data.iloc[:,2].values, data.iloc[:,3].values, c=pd.Categorical(data.iloc[:,-1]).codes, cmap = mpl.colors.ListedColormap(['g', 'r', 'b']))
# cmap参数是对应到不同的y_meshgrid_hat的值的颜色
参考: http://sklearn.apachecn.org/cn/0.19.0/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py

# 类似的绘图方法
x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
cm_light = mpl.colors.ListedColormap(['#FF8080', '#80FF80', '#8080FF', '#F0F080'])
cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b', 'y'])
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
plt.contour(x1, x2, y_test, levels=(0,1,2), colors='k', linestyles='--')
plt.scatter(x[:, 0], x[:, 1], s=20, c=y, cmap=cm_dark, edgecolors='k', alpha=0.7)
plt.xlabel('$X_1$', fontsize=11)
plt.ylabel('$X_2$', fontsize=11)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.grid(b=True)
plt.tight_layout(pad=2.5)
plt.title('SVM多分类方法：One/One or One/Other', fontsize=14)
plt.show()


# 2.单独做legend，不推荐，需要自己去看对应的类别，如果需要绘图上的调整还是用python吧
patchs = [mpatches.Patch(color='red', label='Iris-setosa'),
          mpatches.Patch(color='green', label='Iris-versicolor'),
          mpatches.Patch(color='blue', label='Iris-virginica')]
plt.legend(handles=patchs, fancybox=True, framealpha=0.8)

# 3.对预测值和真实值进行对比
y_order = y_test.argsort()
plt.plot(np.arange(len(y_order)), y_test[y_order], 'g-', label='True')
plt.plot(np.arange(len(y_order)), y_test_hat[y_order], 'r-', label='Predicted')
# 分别绘制真实值和预测值，看两者的差别
```


## 模型保存
```Python
from sklearn.externals import joblib
import os
if os.path.exists('xxx.model'):
	model = joblib.load('xxx.model')
else:
	joblib.dump(model, 'xxx.model')
```




## tips
1. 对于使用sklearn过程中出现warning的情况，可以通过使用下列代码进行warning的隐藏
```Python
import warnings
warnings.filterwarnings('ignore')
```
2. 对于使用jupyter notbook的同学可以使用magic命令使得画图的时候直接显示图片代替 “matplotlib.pyplot.show()”
```Python
% matplotlib inline
```

## 还有一些东西没来得及总计，有机会再补充



<hr />
