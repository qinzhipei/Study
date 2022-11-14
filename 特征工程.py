# -*- coding: utf-8 -*-"
'特征工程'
'''
2.3 数据分级：定类、定序、定量、定比

#3 特征增强：数据清洗
3.1 删除有缺失值的相关行
3.2 填充：fillna填充（定量） pipeline中填充
3.3 标准化、归一化：z分位数归一化 MINMAX标准化 行归一化

#4 特征构建
4.1 定性数据填充
4.2 自定义填充器 定性：继承TransformerMixin 定量：继承SimpleImputer
4.3 定类、定序等级编码；连续特征分箱
4.4 扩展数值特征：特征乘积形成多项式特征
4.5 针对文本的特征构建：CountVectorizer，TF-IDF

#5 特征选择
5.1 基于统计的选择: 基于相关系数、基于假设检验
5.2 基于模型的选择：基于树模型、基于线性模型

#6 特征转换

#7 特征学习
7.1 RBM：使用[概率模型]学习新特征
7.2 词向量：实例化gensim 应用：信息检索

#8 案例分析
8.1 CV：面部识别
8.2 NLP：文本聚类:潜在语义分析
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn.base import TransformerMixin #定类列填充器
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion #并排排列特征
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score # scikit-learn的特征选择模块
from sklearn.feature_selection import SelectKBest,chi2,f_classif #基于假设检验的特征选择
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel #基于线性模型
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC   # 选择的依据
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
import gensim
import logging #日志记录器，查看详细训练过程
from gensim.models import word2vec, Word2Vec
from functools import reduce
from sklearn.datasets import fetch_lfw_people # olivett 数据集
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools 
plt.style.use('fivethirtyeight') #一种可视化风格


'''2.2 数据定量、定性'''
salary_ranges = pd.read_csv(
    'E:/BaiduNetdiskDownload/Feature_Understanding/Salary_Ranges_by_Job_Classification.csv',
    index_col=0)
salary_ranges.head()
salary_ranges.info()
salary_ranges.isnull().sum()
salary_ranges.describe()

salary_ranges = salary_ranges[['Biweekly High Rate', 'Grade']]
salary_ranges.head()

salary_ranges['Biweekly High Rate'].describe()

# 使用map，将函数映射到整个数据集
#删除美元符号
salary_ranges['Biweekly High Rate'] = salary_ranges[
    'Biweekly High Rate'].map(lambda value: value.replace('$', ''))
# 转换成浮点型
salary_ranges['Biweekly High Rate'] = salary_ranges['Biweekly High Rate'].astype(float)
# 将Grade转换成字符串
salary_ranges['Grade'] = salary_ranges['Grade'].astype(str)



''''2.3 数据分级'''
'2.3.1 定类等级'
G_value = salary_ranges['Grade'].value_counts() #使用value_counts()进行计数
G_value.unique().sum()
# 由于数据太多 选取前20名进行可视化
G_value.sort_values(ascending=False).head(20).plot(kind='bar')

'2.3.2 定序等级：数值需要转化为str'
customer = pd.read_csv(r'E:/BaiduNetdiskDownload/Feature_Understanding/2013_SFO_Customer_survey.csv')
customer.head()
art_ratings = customer['Q7A_ART']
art_ratings.describe() # pandas将其视为数值进行处理 但是几位类别 是定性数据中的定序数据
# 如果把0和6删除。剩下5个有序列别类似于餐厅的评分 
art_ratings = art_ratings[(art_ratings>=1) & (art_ratings<=5)]
art_ratings = art_ratings.astype(str)
art_ratings.value_counts().plot(kind='pie')
art_ratings.value_counts().plot(kind='bar')

'2.3.3 定距等级：温度'
climate = pd.read_csv('E:\2022Study\源代码文件\data\GlobalLandTemperaturesByCity.csv')


'2.3.4 定比等级'







'''3 特征增强：数据清洗'''
''''3.1 删除缺失值相关行'''
pima = pd.read_csv(r'E:/2022study/Dataset/diabetes.csv') #印第安人糖尿病数据集
pima.head()
pima.info()
pima['Outcome'].value_counts(normalize=True) #空准确率

col = 'Glucose' #血糖浓度
plt.hist(pima[pima['Outcome']==0][col],10,alpha=0.5,label='non-diabetes') #没有糖尿病的血糖浓度
plt.hist(pima[pima['Outcome']==1][col],10,alpha=0.5,label='diabetes')
plt.legend(loc='upper right')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.title('Histogram of {}'.format(col))
plt.show()

sns.heatmap(pima.corr()) #sns.heatmap：相关矩阵热力图

pima.describe() #发现缺失值被0填充

columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in columns:
    pima[col].replace([0],[None],inplace=True) #将0替换成None，再做缺失值统计
pima.isnull().sum()
pima.head()

pima_dropped = pima.dropna() #存在缺失值的行
pima_dropped.shape #392个缺失值
p1 =pima.mean() #去掉缺失值后，每列的均值
p2 =pima_dropped.mean() #缺失值每列的均值
(p2-p1)/p1 #均值变化百分比

#机器学习
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

X_dropped = pima_dropped.drop('Outcome',axis=1)
y_dropped = pima_dropped['Outcome']

#KNN
knn_params = {'n_neighbors':[1,2,3,4,5,6,7]}
knn = KNeighborsClassifier()

grid = GridSearchCV(knn, knn_params)
grid.fit(X_dropped,y_dropped)
print(grid.best_score_,grid.best_params_) #去掉缺失值后：0.735
#通过网格搜索，找到交叉验证准确率最高的KNN参数


'''3.2 填充'''
empty_glucose_index = pima[pima['Glucose'].isnull()].index
#fillna填充均值
pima['Glucose'].fillna(pima['Glucose'].mean(),inplace=True)

from sklearn.impute import SimpleImputer #填充器
imputer = SimpleImputer(strategy = 'mean') #实例化对象
pima_imputed = pd.DataFrame(imputer.fit_transform(pima))
pima_imputed.head()
pima_imputed.isnull().sum()

X = pima[['Insulin']].copy() #复制数据集
y = pima['Outcome'].copy()
X.isnull().sum()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=99)
#均值填充 注：要用【训练集】的均值填充训练集和测试集
training_mean = X_train.mean()
X_train = X_train.fillna(training_mean)
X_test = X_test.fillna(training_mean)
print(training_mean)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

#零填充
pima_zero = pima.fillna(0)
X_zero_dropped = pima_zero.drop('Outcome',axis=1)
y_zero_dropped = pima_zero['Outcome']
knn_params = {'n_neighbors':[1,2,3,4,5,6,7]}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, knn_params)
grid.fit(X_zero_dropped,y_zero_dropped)
print(grid.best_score_,grid.best_params_)  #零填充：0.741


'在机器学习流水线(pipeline)中填充值'
from sklearn.pipeline import Pipeline 
knn_params = {'classify__n_neighbors':[1,2,3,4,5,6,7]} #classify__n_neighbors:估计器里存在的键
knn = KNeighborsClassifier()
#Pipeline分两步：一个（填充策略为mean/zero的）Imputer,一个knn分类器
mean_impute = Pipeline([('imputer',SimpleImputer(strategy='mean')),('classify',knn)])  #均值填充
mean_impute_median = Pipeline([('imputer',SimpleImputer(strategy='median')),('classify',knn)]) #中位数填充
X_dropped = pima.drop('Outcome',axis=1)
y_dropped = pima['Outcome']
grid = GridSearchCV(mean_impute, knn_params)
grid.fit(X_dropped,y_dropped)
grid2 = GridSearchCV(mean_impute_median, knn_params)
grid2.fit(X_dropped,y_dropped)
print(grid.best_score_,grid.best_params_) #用均值填充：0.732
print(grid2.best_score_,grid2.best_params_) #中位数填充：0.729



''''3.3 标准化、归一化'''
pima_imputed.hist(figsize=(15,15))
pima_imputed.hist(figsize=(15,15),sharex=True) #x轴范围相同

'z分位数归一化'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
glucose_z = scaler.fit_transform(pima[['Glucose']]) #使用双方括号，因为transform需要一个Dataframe
#查看缩放后数据分布
ax = pd.Series(glucose_z.reshape(-1,)).hist() #只变了x轴
ax.set_title('设置标题用set_title')

#z归一化后的各特征分布
scaler = StandardScaler()
pima_imputed_z = pd.DataFrame(scaler.fit_transform(pima_imputed))
pima_imputed_z.hist(figsize=(15,15),sharex=True)

#将StandardScaler插到Pipeline中
knn_params = {'imputer__strategy':['mean','median'],
                                 'classify__n_neighbors':[1,2,3,4,5,6,7]}
#估计器：填充+Z归一化+knn分类
impute_standardized = Pipeline([('imputer',SimpleImputer()),('standardize',StandardScaler()),
                                ('classify',knn)])
X_dropped = pima.drop('Outcome',axis=1)
y_dropped = pima['Outcome']
grid = GridSearchCV(impute_standardized, knn_params)
grid.fit(X_dropped,y_dropped)
print(grid.best_score_,grid.best_params_) #z标准化的流水线：0.754

'MINMAX标准化'
from sklearn.preprocessing import MinMaxScaler
MINMAX = MinMaxScaler()
pima_imputed_MINMAX = pd.DataFrame(MINMAX.fit_transform(pima_imputed))
#估计器：填充+Minmax标准化+knn分类
impute_standardized2 = Pipeline([('imputer',SimpleImputer()),('standardize',MinMaxScaler()),
                                ('classify',knn)])
grid = GridSearchCV(impute_standardized2, knn_params)
grid.fit(X_dropped,y_dropped)
print(grid.best_score_,grid.best_params_) #0.763

'行归一化'
from sklearn.preprocessing import Normalizer
norm = Normalizer()
pima_imputed_norm = pd.DataFrame(norm.fit_transform(pima_imputed))
#估计器：填充+行归一化+knn分类
impute_standardized3 = Pipeline([('imputer',SimpleImputer()),('standardize',Normalizer()),
                                ('classify',knn)])
grid = GridSearchCV(impute_standardized3, knn_params)
grid.fit(X_dropped,y_dropped)
print(grid.best_score_,grid.best_params_) #0.70

'''总结：
3.1 去掉缺失值后：0.735
3.2 零填充：0.741 均值填充：0.732 中位数填充：0.729
3.3 中位数填充+z标准化：0.754 Minmax:0.763 行归一化：0.70'''




'''4 特征构建'''
'''4.1 创造、检查数据集'''
#用DataFrame创建表格数据
X = pd.DataFrame({'city': ['tokyo', None, 'london', 'seattle', 'san francisco', 'tokyo'],
                 'boolean': ['yes', 'no', None, 'no', 'no', 'yes'],
                 'ordinal_column': ['somewhat like', 'like', 'somewhat like', 'like', 
                                   'somewhat like', 'dislike'],
                 'quantitative_column': [1, 11, -.5, 10, None, 20]})
print(X)
#boolean：布尔数据 city:定类 ordinal_column：定序 quantitative_column：定比

#对于定性列city，可计算出【最常见的类别】用于填充
X['city'].fillna(X['city'].value_counts().index[0])


'''4.2 自定义填充器'''
'如果需要填充多个定性类/boolean,需要自定义定性列填充器(class)'
from sklearn.base import TransformerMixin 
class CustomCategoryImputer(TransformerMixin):  
    """继承scikit-learn的TransformerMixin类，它包括一个.fit_transform方法，会调用我们创建
    的.fit和.transform方法。这能让我们的转换器和scikit-learn的转换器保持结构一致。
    """
    def __init__(self, cols=None):
        self.cols = cols #初始化一个实例属性self.cols,即我们需要填充的列

    def transform(self, df):
        X = df.copy() #接收一个Dataframe，并复制
        for col in self.cols:
            X[col].fillna(X[col].value_counts().index[0], inplace=True) #对所有指定列填充最常见类别
        return X

    def fit(self, *_):
        return self

# 在列上应用自定义分类填充器
cci = CustomCategoryImputer(cols=['city', 'boolean'])
cci.fit_transform(X)


'自定义定量列填充器'
# 按名称对列进行转换的填充器
from sklearn.impute import SimpleImputer

class CustomQuantitativeImputer(TransformerMixin):
    def __init__(self, cols=None, strategy='mean'):
        self.cols = cols
        self.strategy = strategy

    def transform(self, df):
        X = df.copy()
        impute = SimpleImputer(strategy=self.strategy)
        for col in self.cols:
            X[col] = impute.fit_transform(X[[col]])
        return X

    def fit(self, *_):
        return self

cqi = CustomQuantitativeImputer(cols=['quantitative_column'], strategy='mean')
cqi.fit_transform(X)

#把定类填充和定量填充放到流水线里
from sklearn.pipeline import Pipeline
imputer = Pipeline([('quant', cqi), ('category', cci)])
imputer.fit_transform(X) #city boolean quantitative_column里都没有缺失值了


'''4.3 编码分类变量'''
'定类等级编码（boolean,city)'
pd.get_dummies(X,columns=['city','boolean'],prefix_sep='__') #prefix_sep:分隔符
# 自定义虚拟变量编码器
class CustomDummifier(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, X):
        return pd.get_dummies(X, columns=self.cols)

    def fit(self, *_):
        return self

cd = CustomDummifier(cols=['boolean', 'city'])
cd.fit_transform(X)

'定序等级编码(ordinal_column):标签编码器'
#单变量编码
# 创建一个列表，顺序数据对应于列表索引
ordering = ['dislike', 'somewhat like', 'like']  # 0是dislike, 1是somewhat like，2是like
# 在将ordering映射到顺序列之前，先看一下列
print(X['ordinal_column'])

# 将ordering映射到顺序列
print(X['ordinal_column'].map(lambda x: ordering.index(x)))
#scikit-learn的LabelEncoder方法虽然也能排序，但是不能按照我们设定的顺序进行编码

# 将自定义标签编码放进流水线
class CustomEncoder(TransformerMixin):
    def __init__(self, col, ordering=None):
        self.ordering = ordering
        self.col = col

    def transform(self, df):
        X = df.copy()
        X[self.col] = X[self.col].map(lambda x: self.ordering.index(x))
        return X

    def fit(self, *_):
        return self

ce = CustomEncoder(col='ordinal_column', ordering=['dislike', 'somewhat like', 'like'])
ce.fit_transform(X)


'连续特征分箱(binning)'
pd.cut(X['quantitative_column'], bins=3)
pd.cut(X['quantitative_column'], bins=3, labels=False)

class CustomCutter(TransformerMixin):
    def __init__(self, col, bins, labels=False):
        self.labels = labels
        self.bins = bins
        self.col = col

    def transform(self, df):
        X = df.copy()
        X[self.col] = pd.cut(X[self.col], bins=self.bins, labels=self.labels)
        return X

    def fit(self, *_):
        return self

cc = CustomCutter(col='quantitative_column', bins=3)
cc.fit_transform(X)
# 现在quantitative_column列处于定序等级，不需要引入虚拟变量

'创建流水线'
from sklearn.pipeline import Pipeline
pipe = Pipeline([("imputer", imputer), ('dummify', cd), 
                 ('encode', ce), ('cut', cc)])
''' 先是imputer填充缺失值 然后是虚拟变量整定类 
    接着是编码整定序 最后分箱整定量 '''

#进流水线前的数据
print(X)
#拟合流水线
pipe.fit(X)
pipe.transform(X) #完成全部编码


'4.4 扩展数值特征'
'4.4.1 根据胸部加速度计识别动作的数据集'
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import GridSearchCV
df = pd.read_csv(r'E:/2022study/Dataset/activity_recognizer.csv')
df.columns = ['index', 'x', 'y', 'z', 'activity']
df.head()

df['activity'].value_counts(normalize=True).sort_values(ascending=False).plot(kind='bar')
df['activity'].value_counts(normalize=True).sort_values(ascending=False).plot(kind='pie')

X = df[['x', 'y', 'z']] #feature
y = df['activity'] #response
knn_params = {'n_neighbors': [3, 4, 5, 6]}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, knn_params)
grid.fit(X, y)
print(grid.best_score_, grid.best_params_)


'将原有的列进行乘积，衍生多项式特征交互'



'''4.5 针对文本的特征构建'''
'长文本包括一系列类别，或称为词项（token）'
from sklearn.feature_extraction.text import CountVectorizer
tweets = pd.read_csv(r'E:/2022study/Dataset/twitter.csv',
                      encoding_errors = 'ignore') #忽视无法解析的字符
X = tweets['text']
y = tweets['sentiment']
vect = CountVectorizer()
_ = vect.fit_transform(X)
print(_.shape) #(100,614) 614个词
vect = CountVectorizer(stop_words='english') #停用english
#停用词是在文本检索过程中出现频率很高但又没有相关内容的单词,如and or in Web 的 嗯 
_ = vect.fit_transform(X)
print(_.shape) #526个词
# min_df:忽略在文档中出现频率低于阈值的词，减少特征的数量
vect = CountVectorizer(min_df=.05)  # 只保留至少在5%文档中出现的单词
_ = vect.fit_transform(X)
print(_.shape) #34个词
vect = CountVectorizer(max_df=.5)  # 只保留之多在50%文档中出现的单词,推断停用词
_ = vect.fit_transform(X)
print(_.shape) #610个
vect = CountVectorizer(ngram_range=(1, 5))  # 最多包括5个单词的短语
_ = vect.fit_transform(X)
print(_.shape)
# 设置分析器作为参数，以判断特征是单词还是短语，默认为单词
vect = CountVectorizer(analyzer='word')
_ = vect.fit_transform(X)
print(_.shape)


'TF-IDF'
'''TF-IDF向量器由两部分组成：表示词频的TF部分，以及表示逆文档频率的IDF部分。
TF：衡量词在文档中出现的频率，一般会对词频进行归一化； IDF：衡量词的重要性，减少常见词的权重，加大稀有词的权重。
TF-IDF是一个用于信息检索和聚类的词加权方法。'''
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer将词项计数按照出现频率进行归一化
vect = CountVectorizer()
_ =  vect.fit_transform(X)
print(_.shape, _[0, :].mean()) #(100, 614) 0.03257328990228014

vect = TfidfVectorizer()
_ = vect.fit_transform(X)
print(_.shape, _[0, :].mean()) #(100, 614) 0.006835734591594493


'在机器学习流水线中使用文本'
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯，特征多时更快
y.value_counts(normalize=True) # neutral  0.61  positive  0.32  negative 0.07
# 创建流水线参数，向量化转换器+分类器
pipe_params = {'vect__ngram_range': [(1, 1), (1, 2)], 'vect__max_features':[1000, 10000], 
              'vect__stop_words': [None, 'english']}
pipe = Pipeline([('vect', CountVectorizer()), ('classifty', MultinomialNB())]) 
grid = GridSearchCV(pipe, pipe_params)
grid.fit(X, y)
print(grid.best_score_, grid.best_params_) #0.71

# 加入TfidfVectorizer进行调优
# FeatureUnion模块，可以水平（并排）排列特征，这样，在一个流水线中可以使用多个类型的文本特征构建器。
from sklearn.pipeline import FeatureUnion
featurizer = FeatureUnion([('tfidf_vect', TfidfVectorizer()), ('count_vect', CountVectorizer())])
_ = featurizer.fit_transform(X)
print(_.shape) #(100, 1228)
# 因为TfidfVectorizer和CountVectorizer并排，所以列数加倍

featurizer.set_params(tfidf_vect__max_features=100, count_vect__ngram_range=(1, 2), 
                     count_vect__max_features=300) # TfidfVectorizer只保留100个单词，而CountVectorizer保留300个1-2个单词的短语
_ = featurizer.fit_transform(X)
print(_.shape) # (100,400)
pipe_params = {'featurizer__count_vect__ngram_range': [(1, 1), (1, 2)], 
              'featurizer__count_vect__max_features': [1000, 10000],
              'featurizer__count_vect__stop_words': [None, 'english'],
              'featurizer__tfidf_vect__ngram_range': [(1, 1), (1, 2)],
              'featurizer__tfidf_vect__max_features': [1000, 10000],
              'featurizer__tfidf_vect__stop_words': [None, 'english']}
pipe = Pipeline([('featurizer', featurizer), ('classify', MultinomialNB())])
grid = GridSearchCV(pipe, pipe_params)
grid.fit(X, y)
print(grid.best_score_, grid.best_params_) #两个向量化器的特征结合：0.72（单个CountVectorizer为0.71）
' CountVectorizer和TF-IDF中，文档只是单词的集合；而词嵌入关注上下文的处理'




'''5 特征选择：对坏属性说不！'''
'''基于统计的特征选择(相关系数，P值)
基于模型的特征选择（基于树模型；基于线性模型和正则化）
来把对response影响小的feature筛掉
'''
'''1.如果特征是[分类]的,那么从SelectKBest开始，用卡方或基于树的选择器。
2.如果特征基本是[定量]的,用线性模型和基于相关性的选择器一般效果更好。
3.如果是[二元分类]问题,考虑使用SelectFromModel和SVC.
4.在手动选择前，探索性数据分析会很有益处。
5.不能低估领域知识的重要性。
'''
#判断信用卡逾期
credit = pd.read_csv(r'E:/2022study/Dataset/credit_card_default.csv')
credit.head(5)
credit.info()
X = credit.iloc[:,:-1]
X = X.drop(['ID'],axis=1)
y=credit.iloc[:,-1] #最后一列
y.value_counts() #0：23364  1：6636
y.value_counts(normalize=True) #空准确率为0.7788

'''5.1 基于统计的选择'''

'5.1.1 基于相关系数的特征选择'
pearson=credit.corr()
pearson['default payment next month'] #与y的相关系数
pearson['default payment next month'][:-1].abs() #去掉y与y本身
index = pearson['default payment next month'][:-1].abs() > 0.1
credit.columns[:-1][index] #与y的相关系数大于0.1的特征
X_subset = X.loc[:, index]

#特征选择前后KNN性能对比
from sklearn.model_selection import cross_val_score
cross_val_score(KNeighborsClassifier(),X,y,cv=5).mean() #0.755
cross_val_score(KNeighborsClassifier(),X_subset,y,cv=5).mean() #0.789,有提升

'5.1.2 基于假设检验的特征选择'
'p值越小，拒绝原假设，认为有关系'
from sklearn.feature_selection import SelectKBest,chi2,f_classif   #卡方检验，f分布
select_model = SelectKBest(score_func=f_classif,k=5)   #k：要选择出的特征个数 f检验：默认，适合分类问题
select_model.fit_transform(X,y)
select_model.get_support() #True的就是留下来的特征
select_model.get_support(indices=True)
select_model.scores_
select_model.pvalues_


'''5.2 基于模型的选择'''
'5.2.1 基于树的模型'
from sklearn.tree import DecisionTreeClassifier
Dct = DecisionTreeClassifier()
Dct.fit(X,y) #这里并不是要用决策树构建模型，而是通过决策树把样本特征的重要性体现出来
Dct.feature_importances_
feature_importances = pd.DataFrame({'feature':X.columns,'important':Dct.feature_importances_}
                                 ).sort_values('important',ascending=False) #按照重要性排序
feature_tree = feature_importances['feature'][:7]  
X_subset_tree = X[feature_tree] #选出来的7个feature索引

#模型评估
from sklearn.ensemble import RandomForestClassifier
#决策树
cross_val_score(DecisionTreeClassifier(), X, y, cv=20).mean() #0.725
cross_val_score(DecisionTreeClassifier(), X_subset_tree, y, cv=20).mean()  #0.696
#随机森林
cross_val_score(RandomForestClassifier(), X, y, cv=20).mean() #0.816
cross_val_score(RandomForestClassifier(), X_subset_tree, y, cv=20).mean()  #0.807
#KNN
cross_val_score(KNeighborsClassifier(), X, y, cv=20).mean()  #0.755
cross_val_score(KNeighborsClassifier(), X_subset_tree, y, cv=20).mean()  #0.739 

'5.2.2 基于线性模型'
from sklearn.feature_selection import SelectFromModel   # 特征选择
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC   # 选择的依据
#LinearSVC实现了线性分类支持向量机，根据liblinear实现的，可以用于二类分类，也可以用于多类分类。
#LinearSVC最小化squared hinge loss，会惩罚截距，One-vs-All方式来多分类；而SVC最小化hinge loss
select_model = SelectFromModel(estimator=LinearSVC(penalty="l2"), threshold='mean') #实例化
#threshold：用于特征选择的阈值。保留mean更高或相等的要素
select_model.fit(X, y)
index = select_model.get_support()
X_subset_linear = X[X.columns[index]]
cross_val_score(RandomForestClassifier(), X_subset_linear, y, cv=20).mean() #0.798

clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
    ('classification', RandomForestClassifier())
])
cross_val_score(clf, X_subset_linear, y, cv=20).mean()
'''特征一旦超过100，通过特征选择就会很麻烦，此时还有降维与特征转换'''


'''6 特征转换'''

'''7 特征学习'''
'''RBM:[无监督]的特征学习算法，使用[概率模型]学习新特征，RBM提取的特征在线性模型中效果最佳。
浅层（2层：可见层+隐藏层）的神经网络；对称二分图
限制：不允许任何层内通信，每个节点可以独立地创造权重和偏差，使得特征之间尽量独立。
'''

'7.1 RBM'
#导入MNIST
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
images = pd.read_csv(r'E:/2022study/Dataset/mnist_train.csv',delimiter=',')
images.shape #(60000, 785)
images_y = images['label']
images_x = images.drop('label',1)
i1 = np.array(images_x.iloc[1]) #第一张图
np.min(images_x), np.max(images_x) #值很大，但sklearn的RBM会进行0-1的缩放
plt.imshow(i1.reshape(28, 28), cmap=plt.cm.gray_r)

'伯努利RBM'
images_x = images_x / 255 #缩放
images_x = (images_x > 0.5).astype(float) #二分黑白
np.min(images_x), np.max(images_x)
i1 = np.array(images_x.iloc[1])
plt.imshow(i1.reshape(28, 28), cmap=plt.cm.gray_r) #图像变清晰了

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(images_x)
# 绘制前100个主成分",
plt.figure(figsize=(10, 10))
for i, comp in enumerate(pca.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r)
        plt.xticks(())
        plt.yticks(())
plt.suptitle('100 components extracted by PCA')
plt.show()

# 前30个特征捕捉64%的信息
pca.explained_variance_ratio_[:30].sum()
#碎石图
full_pca = PCA(n_components=784)
full_pca.fit(images_x)
plt.plot(np.cumsum(full_pca.explained_variance_ratio_))

pca.transform(images_x.iloc[:1]) #提取特征
np.dot(images_x[:1]-images_x.mean(axis=0), pca.components_.T) #矩阵乘法

'从MNIST中提取RBM特征'
# 设置random_state 固定权重和误差
# verbose是True 观看训练
# n_iter是（网络前后向）传递次数
# n_components和PCA和LDA一样 我们希望创建的特征数
# n_components可以是任意整数 小于 等于或大于原始的特征均可
rbm = BernoulliRBM(random_state=0, verbose=True, n_iter=20, n_components=100)
rbm.fit(images_x)

# 可视化RBM特征
plt.figure(figsize=(10, 10))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM')

# 好像有些特征一样 但是其实所有的特征都不一样（虽然有的很类似）
np.unique(rbm.components_.mean(axis=1)).shape #100

#用RBM转换数字
image_new_features = rbm.transform(images_x.iloc[:1]).reshape(100,)
image_new_features

# 不是简单的矩阵乘法
# 是神经网络 几个矩阵乘法 转换特征
np.dot(images_x[:1]-images_x.mean(axis=0), rbm.components_.T)

# 从第一个图像上提取20个最有代表性的特征
top_features = image_new_features.argsort()[-20:][::-1]
print(top_features) #这些特征都可以达到100%的RBM
image_new_features[top_features]








'7.2 词向量'
#Word2Vec
import gensim
import logging #日志记录器，查看详细训练过程
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import word2vec, Word2Vec
sentences = word2vec.Text8Corpus('E:/2022Study/源代码文件/data/text8')

# 实例化gensim模块
# min-count是忽略次数比它小的词
# size是要学习的词的维数
model = gensim.models.Word2Vec(sentences, min_count=1, vector_size=20)
# 单个词的嵌入
model.wv['fuck']
# woman + king - man = queen
model.wv.most_similar(positive=['woman', 'fuck'], negative=['man'], topn=10)

#Google预训练 300万个单词库
model = gensim.models.KeyedVectors.load_word2vec_format(r'E:\2022Study\Dataset\GoogleNews-vectors-negative300.bin', 
                                                        binary=True)
len(model.key_to_index) #3000000

model.most_similar(positive=['woman','penis'], negative=['vagina'],topn=10)
model.doesnt_match("duck bear cat tree".split()) #不属于同类别的单词
model.similarity('python','java')


'词嵌入的应用：信息检索'
# 从gensim中获取单词的词嵌入 没有则返回None
def get_embedding(string):
    try: #如果model.wv里有这个词，返回词嵌入
        return model.wv[string]
    except:
        return None
    
#创建3个文章标题，一个关于狗，一个关于猫，一个干扰项
sentences2 = ["this is about a dog","this is about a cat","this is about nothing"]
#《数据科学原理》章名列表
sentences = """How to Sound Like a Data Scientist
Types of Data
The Five Steps of Data Science
Basic Mathematics
A Gentle Introduction to Probability
Advanced Probability
Basic Statistics
Advanced Statistics
Communicating Data
Machine Learning Essentials
Beyond the Essentials
Case Studies """.split('\n')

model = gensim.models.Word2Vec(sentences, min_count=1, vector_size=12)

from functools import reduce
#对每个句子创建一个3x300的向量化矩阵
# 3x300的零矩阵
vectorized_sentences = np.zeros((len(sentences),300))
# 每个句子
for i, sentence in enumerate(sentences):
    # 分词
    words = sentence.split(' ')
    # 进行【词嵌入】
    embedded_words = [get_embedding(w) for w in words]
    embedded_words = filter(lambda x:x is not None, embedded_words) #过滤:filter()
    # 对标题进行矢量化 取均值
    vectorized_sentence = reduce(lambda x,y:x+y, embedded_words)/len(list(embedded_words))
    vectorized_sentences[i:] = vectorized_sentence #输出向量化矩阵
#报错了
vectorized_sentences.shape #(12,300)

#向量化后，可以对句子和参考词求点积，进行比较
reference_word = 'AI' #给定词AI，输出与这个词最相关的3个的句子
best_sentence_idx = np.dot(vectorized_sentences, 
                           get_embedding(reference_word)).argsort()[-3:][::-1] #【词嵌入和向量化矩阵的乘积】
 



'''8 案例分析'''
'8.1 面部识别'
'数据加载'
#人脸识别数据集
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape #样本数，宽，高
n_samples, h, w #(1288, 50, 37)

X = lfw_people.data
y = lfw_people.target
n_features = X.shape[1]

n_features #50*37=1850
X.shape #(1288, 1850)

'数据探索'
# 画一张脸
plt.imshow(X[600].reshape((h, w)), cmap=plt.cm.gray)
lfw_people.target_names[y[600]] #人名

# 人脸缩放（标准化）
plt.imshow(StandardScaler().fit_transform(X)[600].reshape((h, w)), cmap=plt.cm.gray)
lfw_people.target_names[y[600]]

# 预测图片是谁
target_names = lfw_people.target_names
n_classes = target_names.shape[0] #7种（一共7个人）
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

'用Pipeline建模'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

pca = PCA(n_components=200, whiten=True)
preprocessing = Pipeline([('scale', StandardScaler()), ('pca', pca)]) # 创建流水线 进行扩充 然后PCA
#从966张脸中提取200个eigenfaces

preprocessing.fit(X_train) # 在训练集上拟合数据
extracted_pca = preprocessing.steps[1][1] # 从流水线上取PCA
plt.plot(np.cumsum(extracted_pca.explained_variance_ratio_)) #碎石图

#创建函数，绘制PCA主成分
comp = extracted_pca.components_
image_shape = (h, w)
def plot_gallery(title, images, n_col, n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,            
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()  

plot_gallery('PCA componenets', comp[:16], 4,4) #绘制人脸特征

import itertools
#显示混淆矩阵，包括热标签和归一化选项
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#不用PCA的线性模型预测raw data
from sklearn.linear_model import LogisticRegression
t0 = time()
logreg = LogisticRegression()
param_grid = {'C': [1e-2, 1e-1,1e0,1e1, 1e2]}
clf = GridSearchCV(logreg, param_grid)
clf = clf.fit(X_train, y_train)
best_clf = clf.best_estimator_

# 用测试集预测姓名
y_pred = best_clf.predict(X_test)
print(accuracy_score(y_pred, y_test), "Accuracy score for best estimator")
print(classification_report(y_test, y_pred, target_names=target_names))
print(plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=range(n_classes)), target_names))
print(round((time() - t0), 1), "seconds to grid search and predict the test set")
#0.82 Acc

# 用PCA 看时间变化
t0 = time()
face_pipeline = Pipeline(steps=[('PCA', PCA(n_components=200)), ('logistic', logreg)])
pipe_param_grid = {'logistic__C': [1e-2, 1e-1,1e0,1e1, 1e2]}
clf = GridSearchCV(face_pipeline, pipe_param_grid)
clf = clf.fit(X_train, y_train)
best_clf = clf.best_estimator_

# 用测试集预测姓名
y_pred = best_clf.predict(X_test)
print(accuracy_score(y_pred, y_test), "Accuracy score for best estimator")
print(classification_report(y_test, y_pred, target_names=target_names))
print(plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=range(n_classes)), target_names))
print(round((time() - t0), 1), "seconds to grid search and predict the test set")
#0.80 ACC 准确率下降了！怎么回事？？？？


# 用grid search寻找最佳模型和准确率
def get_best_model_and_accuracy(model, params, X, y):
    grid = GridSearchCV(model,           # 网格搜索的模型
                        params,          # 搜索的参数
                        error_score=0.)  # 如果出错 正确率是0
    grid.fit(X, y)           # 拟合模型和参数
    # 经典性能参数
    print("Best Accuracy: {}".format(grid.best_score_))
    # 最佳精度的最佳参数
    print("Best Parameters: {}".format(grid.best_params_))
    # 平均拟合数据的时间（秒）
    print("Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3)))
    # 平均预测数据的时间（秒）
    print("Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3)))


# 网格搜索的大型流水线
face_params = {'logistic__C':[1e-2, 1e-1, 1e0, 1e1, 1e2], #正则化强度的倒数
               'preprocessing__pca__n_components':[100, 150, 200, 250, 300],
               'preprocessing__pca__whiten':[True, False],
               'preprocessing__lda__n_components':range(1, 7)  
               # [1, 2, 3, 4, 5, 6] recall the max allowed is n_classes-1
              }
pca = PCA()
lda = LinearDiscriminantAnalysis()

#Pipeline：缩放+PCA模块+LDA模块
preprocessing = Pipeline([('scale', StandardScaler()), ('pca', pca), ('lda', lda)])

logreg = LogisticRegression() #线性分类器

#Pipeline：预处理+分类
face_pipeline = Pipeline(steps=[('preprocessing', preprocessing), ('logistic', logreg)])

#调用参数，得到精度最佳的模型、模型参数和拟合的时间
get_best_model_and_accuracy(face_pipeline, face_params, X, y)
'''Best Accuracy: 0.8547974542273702
Best Parameters: 
{'logistic__C': 1.0, 
'preprocessing__lda__n_components': 6, 
'preprocessing__pca__n_components': 150, 'preprocessing__pca__whiten': False}
Average Time to Fit (s): 0.259
Average Time to Score (s): 0.009
'''












'8.2 文本聚类：潜在语义分析··'



