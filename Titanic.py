# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import re
from time import time
import warnings
warnings.filterwarnings('ignore')
warnings.warn("deprecated", DeprecationWarning)  
from imblearn.pipeline import make_pipeline #pip install imblearn --user

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score,roc_curve,roc_auc_score
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.ensemble import BaggingClassifier #Bagging
from sklearn.ensemble import RandomForestClassifier #随机森林
import xgboost
from xgboost import XGBClassifier
#import lightgbm as lgb


'''1 加载数据集'''
train = pd.read_csv(r'E:\2022Study\Data\Titanic\train.csv')
test = pd.read_csv(r'E:\2022Study\Data\Titanic\test.csv')

train.head() #11个特征
train.info() #age embarked cabin有缺失值
train.isnull().sum()

'合并数据集'
full = train.append(test,ignore_index =True)
full.info()

'''2 数据分析'''
'2.1 男女生存率'
women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women) #0.742

men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men) #0.189

'2.2 swarmplot：人口结构图'
plt.title("Age of people who survived or did not survive")
sns.swarmplot(x = train['Survived'], y=train['Age'])

'2.3 生存情况与人员的船费以及年龄间的散点图'
plt.title("Survived and not survived people with their Fare and Age")
sns.scatterplot(x=train['Age'], y=train['Fare'], hue=train['Survived'])




'''3 特征工程'''
'3.1 特征填充：用Unknown填充Cabin 用中位数填充Age 用均值填充Fare 用众数填充Embarked'
#age：263条缺失 cabin：1013缺失 Fare：1缺失 Embarked：2缺失
full['Age'] = full['Age'].fillna(full['Age'].median())  
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])  #mode:众数
full['Cabin'] = full['Cabin'].fillna("Unknown")  
full['Fare'] = full['Fare'].fillna(full['Fare'].mean()) 

'给Age分箱'
age = full['Age']
age_bin = pd.cut(age,4) #年龄分为4类
Counter(age_bin)


'3.2 特征合并：sibsp和Parch合并为家眷数'
full['Family'] = full['SibSp'] + full['Parch']
full.head()
full = full.drop(['SibSp','Parch'],axis=1)


'3.3 特征提取：映射、哑变量、提取首字母'
'3.3.1 将sex映射为数值'
Sexdict = {'male':1,'female':0}
full['Sex'] = full['Sex'].map(Sexdict)

'3.3.2 特征提取：对登船港口Embarked,客舱等级Pclass,年龄Age做哑变量处理'
embarked = full['Embarked']
embarkedDf = pd.get_dummies(embarked,prefix = 'Embarked') #prefix：前缀
embarkedDf.head()

Pclass = full['Pclass']
PclassDf = pd.get_dummies(Pclass,prefix = 'Pclass') #prefix：前缀
PclassDf.head()

age_binDf = pd.get_dummies(age_bin,prefix = 'Age')

full = pd.concat([full,embarkedDf,PclassDf,age_binDf],axis = 1) #把哑变量列加到数据集里
full = full.drop(['Embarked','Pclass','Age'],axis=1)#把原embarked,pclass列删掉
full.info()

'3.3.3 提取客舱号首字母'
Cabin = full['Cabin']
Cabin = Cabin.map(lambda c:c[0]) #提取首字母
CabinDf = pd.get_dummies(Cabin,prefix = 'Cabin')
CabinDf = CabinDf.drop(['Cabin_U'],axis=1) #把Unknown去掉

full = pd.concat([full,CabinDf],axis=1)
full = full.drop(['Cabin'],axis=1)
full.info()


'3.4 特征选择'
'直接删除'
full = full.drop(['PassengerId','Name','Ticket'],axis=1)
full.info()

'与Survive的相关系数'
corrDf = full.corr()
#绘图
sns.heatmap(corrDf, cmap="YlGnBu")

#各特征与Survive的corr,降序排列
corrDf['Survived'].sort_values(ascending=False) #ascending=False：降序排列
full.head()

'建模得到结果后，对特征作优化尝试'
full2 = full.drop(['Family'],axis=1) #删除corr小的特征

'3.5 特征提取：降维'
full_X = full.drop(['Survived'],axis=1)
pca = PCA(n_components=6)
full3 = pca.fit(full_X)
var = pca.explained_variance_ratio_

#累计方差：碎石图
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
plt.plot(var1)



'''4 构建模型'''
'4.1 建立特征和标签；建立Train,Test和Pred'
#没删family
full_X = full.drop(['Survived'],axis=1)
full_Y = full['Survived']

#删了family
#full_X = full2.drop(['Survived'],axis=1)
#full_Y = full2['Survived']

sourceRow = 891 #原始数据有891行

#训练数据
Source_X = full_X.loc[0:sourceRow-1,:]
Source_Y = full_Y.loc[0:sourceRow-1]
#预测数据
Pred_X = full_X.loc[sourceRow:,:]

Source_X.shape #(891,21)
Source_Y.shape
Pred_X.shape #(418,21)

train_x, test_x, train_y, test_y = train_test_split(Source_X,Source_Y,
                                                    train_size=0.75)
train_x.shape #(668,21)
test_x.shape



'4.2 选择模型：二分类问题'
model = []
'4.2.1 逻辑回归'
logreg = LogisticRegression(solver='liblinear') #有报错，改了求解器
logreg.fit(train_x, train_y)
model.append(logreg)

#网格搜索
param_grid = {'C': [1e-2, 1e-1,1e0,1e1, 1e2]}
clf = GridSearchCV(logreg, param_grid)
clf = clf.fit(train_x, train_y)
best_clf = clf.best_estimator_
model.append(best_clf)

'4.2.2 Bagging,随机森林'
#训练集有放回采样，在多个训练子集上用相同的算法
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),n_estimators = 500,
    max_samples=100,bootstrap=True,n_jobs=-1,oob_score=True)
#n_estimators=500:500个相同的决策器 默认为10
#max_samples=100，表示在数据集上有放回采样 100 个训练实例。
#n_jobs=-1:使用所有空闲核
#oob_score=True，表示包外评估，设定37%左右的实例是未被采样的，用这些实例来对模型进行检验
bag_clf.fit(train_x, train_y)
model.append(bag_clf)

#随机森林
rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16, 
                                 max_depth=15,max_features='sqrt',
                                 n_jobs=-1) 
#500棵树，深度最多为15，每棵树最多16个叶结点（预剪枝）
#max_features:选取的特征子集中的特征个数，可取sqrt,auto,log2
rnd_clf.fit(train_x, train_y)
model.append(rnd_clf)

'4.2.3 XGBoost'
t0 = time()
xgb = XGBClassifier(max_depth=2, #max_depth: 树的深度，默认值是6，值过大容易过拟合
                   learning_rate=0.05, 
                   silent=True,
                   reg_lambda = 1.1, #L2正则化
                   objective='binary:logistic')

#Grid Search+CV 将交叉验证和网格搜索封装在一起
param_test = {'n_estimators': range(10, 100, 1)} #需要优化的参数：基学习器的个数（1-100，函数默认值是100）
xgb_clf = GridSearchCV(estimator = xgb, 
                       param_grid = param_test, 
                       verbose=True, #输出训练过程
                       scoring= 'roc_auc',           
                       cv=5,                   
                       n_jobs=-1,
                       )# 5折交叉验证
#报错：feature_names may not contain [, ] or <
#Debug:import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
train_x.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train_x.columns.values]

xgb_clf.fit(train_x, train_y,eval_metric=["error", "logloss"])
t1 = time()
print(t1-t0) #7秒
model.append(xgb_clf)


'4.2.4 Stacking融合'
'''
关键词：集成框架 准而不同
传统的模型融合,最容易想到的就是对最终结果取平均(算术/平均等)
而stacking将其理解为一种【模型集成框架】,可以对多个【准而不同】的基模型
(比如randomForest,Adaboost,svm...等)进行有效的融合
'''
#5折CV
kf = KFold(n_splits=5,shuffle=True,random_state=42) 

#Ⅰ 构建sklearn中模型的功能类:初始化参数，然后train,pred
class SklearnWrapper(object):
    def __init__(self,clf,seed=0,params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
    def fit(self, x, y):
        return self.clf.fit(x, y)
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    def predict(self,x):
        return self.clf.predict(x)
    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)
    def predict_proba(self, x):
        return self.clf.predict_proba(x)[:, 1]
    def decision_function(self,x):
        return self.clf.decision_function(x)

ntrain = train.shape[0] #train data：891行
ntest = test.shape[0] #test data:418行
    
#Ⅱ 封装交叉验证函数，针对的是【一个】基学习器（clf）
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))  #1*891
    oof_test = np.zeros((ntest,)) #1*418
    oof_test_skf = np.empty((5, ntest))  #5*418的空预测矩阵，待填充
    for i, (train_index, test_index) in enumerate(kf.split(x_train)): #循环5次
       #每一次fold,把891行大train分为713行小train（包含x和y）和178行小test
       x_tr = x_train.iloc[train_index] #小train 
       y_tr = y_train.iloc[train_index]
       x_te = x_train.iloc[test_index] #小test
       clf.fit(x_tr, y_tr) #用clf训练小train
       oof_train[test_index] = clf.predict(x_te) #预测小test
       #预测大test，每走一次fold填充一行，一共走5次，得到5*418的预测值矩阵      
       oof_test_skf[i, :] = clf.predict(x_test)  #5*418
    oof_test[:] = oof_test_skf.mean(axis=0)  #axis=0,按列求平均，最后保留一行1*418
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) #1*418转为418*1（418行）
    #最后返回【这个】模型的训练值矩阵（891*1）和预测值矩阵（418*1）

'''假设第一层有3个模型(P1，P2, P3)，这样你就会得到：
来自5-fold的预测值矩阵 890 X 3和来自Test Data预测值矩阵 418 X 3
第二层：来自5-fold的预测值矩阵 890 X 3 作为你的Train Data，训练第二层的模型
来自Test Data预测值矩阵 418 X 3 就是你的Test Data
'''

#Ⅲ 定义第一层模型和第二层模型.尽量要选择准确又具有很大差异性的模型进行第二层的输入
#第一层：xgboost，随机森林【能搞】，lasso, ridge, logreg【不能】
def stage_1_model():
    #xgboost
    xgb_params = {'max_depth': 5
        , 'learning_rate': 0.05
        , 'n_estimators': 60
        ,'colsample_bytree':0.7
        ,'min_child_weight':5
        , 'n_jobs': 8
        }
    #随机森林
    rf_params = {
        'n_estimators': 100,
        'max_features': 0.2, #?
        'max_depth': 12,
        'min_samples_leaf': 2,
        }
    
    xgb = SklearnWrapper(clf=XGBClassifier, seed=2000, params=xgb_params)
    rf = SklearnWrapper(clf=RandomForestClassifier, seed=2000, params=rf_params)
    #rd = SklearnWrapper(clf=Ridge, seed=2000, params=rd_params)
    #ls = SklearnWrapper(clf=Lasso, seed=2000, params=ls_params)
    
    #得到891*1，418*1
    xgb_oof_train, xgb_oof_test = get_oof(xgb,train_x, train_y, Pred_X) #报错，缺后面三个参数
    rf_oof_train, rf_oof_test = get_oof(rf,train_x, train_y, Pred_X)
    #rd_oof_train, rd_oof_test = get_oof(rd,train_x, train_y, Pred_X)
    #ls_oof_train, ls_oof_test = get_oof(ls,train_x, train_y, Pred_X)
    
    return rf_params,xgb_params #log_params #ls_params,rd_params

def stage_2_model():
    xgb_params_stage2 = {'max_depth': 5
        , 'learning_rate': 0.05
        , 'n_estimators': 50
        ,'colsample_bytree':0.6
        ,'min_child_weight':10
        , 'n_jobs': 8
            } #Xgboost
    rf_params_stage2 = {
        'n_estimators': 50,
        'max_features': 0.2, #?
        'max_depth': 10,
        'min_samples_leaf': 2,
        } #随机森林
    #log_params_stage2 = {'solver':'liblinear'} #逻辑回归
    #ls_params_stage2 = {'alpha':0.005}   #Lasso
    #rd_params_stage2 = {'alpha':20} #Ridge
    
    xgb_stage2 = SklearnWrapper(clf=XGBClassifier, seed=1000, params=xgb_params_stage2)
    rf_stage2 = SklearnWrapper(clf=RandomForestClassifier, seed=1000, params=rf_params_stage2)
    #log_stage2 = SklearnWrapper(clf=LogisticRegression, seed=1000, params=log_params_stage2)
    #rd_stage2 = SklearnWrapper(clf=Ridge, seed=1000, params=rd_params_stage2)
    #ls_stage2 = SklearnWrapper(clf=Lasso, seed=1000, params=ls_params_stage2)
    
    return xgb_params_stage2,rf_params_stage2 #log_stage2 #rd_params_stage2,ls_params_stage2

#Ⅳ Stacking
if __name__=="__main__":
    print("-------->>>>>第一层的基模型<<<<<--------")
    rf_params,xgb_params = stage_1_model()
    xgb = SklearnWrapper(clf=XGBClassifier, seed=2000, params=xgb_params)
    rf = SklearnWrapper(clf=RandomForestClassifier, seed=2000, params=rf_params)
    #logreg = SklearnWrapper(clf=LogisticRegression, seed=2000, params=log_params)
    #rd = SklearnWrapper(clf=Ridge, seed=2000, params=rd_params)
    #ls = SklearnWrapper(clf=Lasso, seed=2000, params=ls_params)
    
    print("-------->>>>>第一阶段训练oof<<<<<--------")
    #返回各模型的训练值矩阵(891, 1)和预测值矩阵(418, 1)
    xgb_oof_train, xgb_oof_test = get_oof(xgb, train_x, train_y, Pred_X)
    rf_oof_train, rf_oof_test = get_oof(rf, train_x, train_y, Pred_X)
    #log_oof_train, log_oof_test = get_oof(logreg, train_x, train_y, Pred_X)
    #rd_oof_train, rd_oof_test = get_oof(rd, train_x, train_y, Pred_X)
    #ls_oof_train, ls_oof_test = get_oof(ls, train_x, train_y, Pred_X)
    
    #得到第一层来自5-fold的预测值矩阵 891*模型数 和 来自Test Data（Pred_X）预测值矩阵 418*模型数
    x_train = np.concatenate((xgb_oof_train, rf_oof_train),
                             axis=1) #来自5-fold的预测值矩阵
    x_test = np.concatenate((xgb_oof_test, rf_oof_test), 
                            axis=1) #来自Test Data（Pred_X）预测值矩阵
    print(x_train.shape,x_test.shape)
    print("Training Stage_1 is complete")
    xgb_params_stage2,rf_params_stage2 = stage_2_model()
    print("Training is complete")

    print("------第二阶段训练开始--------")
    predictions_stage2 = []
    use_vote = True
    xgb_stage2 = SklearnWrapper(clf=XGBClassifier, seed=1000, params=xgb_params_stage2)
    rf_stage2 = SklearnWrapper(clf=RandomForestClassifier, seed=1000, params=rf_params_stage2)
   # log_stage2 = SklearnWrapper(clf=LogisticRegression, seed=1000, params=log_params_stage2)
   # rd_stage2 = SklearnWrapper(clf=Ridge, seed=1000, params=rd_params_stage2)
   # ls_stage2 = SklearnWrapper(clf=Lasso, seed=1000, params=ls_params_stage2)
    for model_two_stage in [xgb_stage2, rf_stage2]:
        model_two_stage.fit(x_train, Source_Y) #x_train(891*4) Source_Y:(891,)
        predictions = model_two_stage.predict_proba(x_test)
        predictions_stage2.append(predictions) #len(predictions_stage2):2

    predictions_avg = sum(predictions_stage2) / len(predictions_stage2)  #融合各模型后每个Pred_Y的可能性概率值
    len(predictions_avg) #418
    if use_vote:
        votes = [(pre > 0.5).astype(int) for pre in predictions_avg]
Counter(votes)
    
'''报错1：None of [Int64Index……are in the [columns]
改：get_oof中x_train改为x_train.iloc
报错2：'SklearnWrapper' object has no attribute 'fit'
改：增加SklearnWrapper中def fit 
报错3：could not broadcast input array from shape (223) into shape (418)
改：stage_1_model中改get_oof(rf,train_x, train_y, test_x)改test_x为Pred_X
报错4：too many values to unpack (expected 3)
改：调用stage_1_model时左边参数个数小于return的参数个数
报错5：model_two_stage.predict_proba(x_test)[:, 1]
      提示：'Ridge','Lasso' object has no attribute 'predict_proba'
改：
报错6：logreg无法融合
改：logreg使用概率函数判断：decision_function大于0正/小于0负; xgb,rf使用大于0.5正，小于0.5负
    （给logreg+0.5可否解决问题？）
报错7：模型全部预测为0
改：
'''


'4.3 模型评估'
len(model) #一共几个模型
predtest = []
predtest_y1 = best_clf.predict(test_x)
predtest.append(predtest_y1)
predtest_y2 = logreg.predict(test_x)
predtest.append(predtest_y2)
predtest_y3 = bag_clf.predict(test_x)
predtest.append(predtest_y3)
predtest_y4 = rnd_clf.predict(test_x)
predtest.append(predtest_y4)
predtest_y5 = xgb_clf.predict(test_x)
predtest.append(predtest_y5)

'K折交叉验证'
#cross_val_score：每次迭代的交叉验证评分，需要取均值
def K_Fold_Score(model):
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X=Source_X, scoring='accuracy',
                             y=Source_Y, cv=10, n_jobs=1)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
    return (np.mean(scores))

K_Acc = []
for i in range(len(model)):
    K_Acc.append(K_Fold_Score(model[i])) #xgb报错

'ACC F1 混淆矩阵 AUC K折ACC'
Acc = []
F1 = []
Conf = []
AUC = []

for i in range(len(predtest)):
    Acc.append(accuracy_score(predtest[i], test_y)) #精度
    F1.append(f1_score(predtest[i], test_y)) #f1
    Conf.append(confusion_matrix(predtest[i], test_y))
    AUC.append(roc_auc_score(predtest[i], test_y))
print(Acc,F1,AUC,K_Acc)


'决策函数，ROC曲线绘制'
#逻辑回归，随机森林，xgboost比较
#.decision_function/.predict_proba:返回一个Numpy数组
#其中每个元素表示【分类器对x_test的预测样本是位于超平面的右侧还是左侧】，以及离超平面有多远。
log_score = logreg.decision_function(test_x)
rnd_score = rnd_clf.predict_proba(test_x) 
xgb_score = xgb_clf.predict_proba(test_x)

fpr1, tpr1, thresholds = roc_curve(test_y,log_score)
fpr2, tpr2, thresholds = roc_curve(test_y,rnd_score[:,1])
fpr3, tpr3, thresholds = roc_curve(test_y,xgb_score[:,1])

plt.plot(fpr1,tpr1,label="Logreg")
plt.plot(fpr2,tpr2,label="Random Forest")
plt.plot(fpr3,tpr3,label="Xgboost")
plt.legend(loc="lower right")
plt.plot()

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()



'5 进行预测'
'保留特征family'
#预测 标签转为int
pred_y0 = np.zeros((len(Pred_X),)) #全预测为0，空准确率
pred_y1 = logreg.predict(Pred_X).astype(int) #LR 0.756
pred_y2 = best_clf.predict(Pred_X).astype(int) #LR+GS 0.756
pred_y3 = bag_clf.predict(Pred_X).astype(int) #Bagging 0.778
pred_y4 = rnd_clf.predict(Pred_X).astype(int) #RF 0.778
pred_y5 = xgb_clf.predict(Pred_X).astype(int) #xgb 0.785
pred_y6 = np.array(votes)

'去掉特征family'
pred_y1b = logreg.predict(Pred_X).astype(int) #LR 
pred_y2b = best_clf.predict(Pred_X).astype(int) #LR+GS 0.770
pred_y3b = bag_clf.predict(Pred_X).astype(int) #Bagging 0.772
pred_y4b = rnd_clf.predict(Pred_X).astype(int) #RF 
pred_y5b = xgb_clf.predict(Pred_X).astype(int) #xgb 0.787

'输出结果'
#生成结果
test.info()
#乘客ID
PassengerId = test['PassengerId']
predDf = pd.DataFrame(
    {'PassengerID':PassengerId,
     'Survived':pred_y0})
predDf.info()
predDf.head(10)

#保存到csv
predDf.to_csv(r'E:\2022Study\Data\Titanic\pred.csv',index=False)

len("hello world")
















