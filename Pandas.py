import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # use Seaborn styles




'''1 Pandas对象'''
'1.1 Series对象：带索引的一维数组'
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data.values
data.index
#自定义索引
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
#series也是特殊的字典，用字典类型创建一个series对象
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
#数据切片
population['California':'New York']

'1.2 DataFrame对象'
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
#构建Df
#字典形式
states = pd.DataFrame({'population': population,
                       'area': area})
states.column
states.index

#series对象形式
pd.DataFrame(population,columns=['population'])
#numpy数组形式
pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])
#结构化数组
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
pd.DataFrame(A)

'1.3 index对象'
ind = pd.Index([2, 3, 5, 7, 11])
print(ind.size, ind.shape, ind.ndim, ind.dtype) #5 (5,) 1 int64

#将index看成有序集合
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB,indA | indB,indA ^ indB



'''2 数据取值与选择'''
'2.1 Series数据选择'
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
'a' in data
data.keys()
list(data.items()) #('a', 0.25), ('b', 0.5), ('c', 0.75), ('d', 1.0)

#索引器
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data
#显式操作
data.loc[1] #'a'
data.loc[1:3] #'a' 'b'
#隐式操作
data.iloc[0] #'a'
data.iloc[0:2] #'a' 'b'

'2.2 DataF数据选择'
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data.values
data.T
data.iloc[:3, :2] #隐式行到3，列到2
data.loc[:'Illinois', :'pop'] #显式
#.ix：混合效果 已被弃用

#筛选
data['Florida':'Illinois']
data.loc[data.density > 100, ['pop', 'density']]
data[data.density > 100] #直接过滤



'''3 Pandas数值计算'''
'3.1 保留索引：使用Numpy通用函数'
#创建series和df
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
print(ser)
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
print(df)

np.exp(ser) #使用Numpy通用函数计算，会保留索引
np.sin(df * np.pi / 4)


'3.2 索引对齐'
'series索引对齐'
#美国面积最大的3各州
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
#美国人口最多的3个州
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
#计算，会输出并集
population / area
'''Alaska              NaN
California    90.413926
New York            NaN
Texas         38.018740'''

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B #0    NaN 1    5.0 2    9.0 3    NaN
A.add(B, fill_value=0) #0    2.0 1    5.0 2    9.0 3    5.0

'DF索引对齐'
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
A + B
print(A.shape,B.shape,(A+B).shape) #(2, 2) (3, 3) (3, 3)

#填充值
fill = A.stack().mean() #.stack() 将二维数组压缩成一维
A.add(B, fill_value=fill)

'DF与Series运算'
#默认按行运算
A = rng.randint(10, size=(3, 4))
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]

#按列计算：axis=0
df.subtract(df['R'], axis=0)

halfrow = df.iloc[0, ::2]
print(df,halfrow)
df-halfrow



'4 处理缺失值：null NaN NA'
'None:Python对象类型的缺失值，只能用于object数组类型'
vals1 = np.array([1, None, 3, 4])
vals1.sum() #报错，'int' and 'NoneType'不能做加法

'NaN:float64浮点数类型，任何系统都兼容'
vals2 = np.array([1, np.nan, 3, 4]) 
vals2.dtype #dtype('float64')
print(vals2.sum(), 1 + np.nan) #nan nan "同化"
#忽略缺失值影响
print(np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2))

'Pandas中None与NaN的差异'
x = pd.Series([1, np.nan, 2, None])
x[0] = None #会自动将none转化为Nan
x

'处理缺失值'
'series上'
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()]
data.dropna()
print(data)
print(data.fillna("haha"))
print(data.fillna(method='ffill')) #向前填充
print(data.fillna(method='bfill')) #向后填充

'df上'
#删除
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
print(df)
df.dropna() #默认，哪行有缺失值，就删除这整行
df.dropna(axis='columns') #哪列有缺失值，就删除这整列
df[3] = np.nan
df.dropna(axis='columns', how='all') #how='all'：只删除所有都是确实值的列
df.dropna(axis='rows', thresh=3) #thresh：设置行或列中非缺失值的最小数量

#填充
print(df.fillna(method='ffill',axis=0) )
print(df.fillna(method='bfill',axis=0))
print(df.fillna(method='ffill',axis=1) )
print(df.fillna(method='bfill',axis=1))



'5 层级索引'
'使用层级索引，可以用series或df表示高维数据'
'5.1 多级索引Series'
#分析美国各州在【不同年份】的人数
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)


#创建多级索引:.MultiIndex
index = pd.MultiIndex.from_tuples(index)
print(index,type(index))

#索引重置 .reindex
pop = pop.reindex(index)
print(pop)
pop[:, 2010] #使用第二个索引

#转化成dataframe：unstack()
pop_df = pop.unstack()
print(pop_df)
pop_df.stack() #转回来

#用df展示高维数据
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})

'5.2 创建多级索引'
#将Index设置为多维即可
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], #Index为二维数组
                  columns=['data1', 'data2'])

#显式创建多级索引 以下效果相同
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])

#添加索引名
pop.index.names = ['state', 'year']
print(pop)

#多级行列索引
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

health_data = pd.DataFrame(data, index=index, columns=columns)
health_data #创建四维数据
health_data.loc[2013] #按行查第一级索引

'5.3 多级索引取值与切片'
#series多级索引
pop['California', 2000] #单个元素获取
pop[:, 2000] #取二级索引

health_data['Guido', 'HR'] #按列索引
health_data.iloc[:2, :2] #行列索引

#先对索引按字典顺序排序，才能用切片
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
print(data) #顺序不对

#sort_index()排序
data = data.sort_index()
print(data)
data['a':'b']

#unstack()
pop.unstack(level=0) 
pop.unstack(level=1)

'5.4 设置与重置索引'
pop_flat = pop.reset_index(name='population')
print(pop_flat) #展开，设置索引
pop_flat.set_index(['state', 'year']) #重置索引

'5.5 聚合函数：level='
#行索引作数据累计
print(health_data.mean(level='year'),
health_data.sum(level='year')) 
#列索引数据累计
health_data.mean(level='year').mean(axis=1, level='type')



'6 合并数据集'
'6.1 Concat'
#定义一个能创建Dataframe某种形式的函数
def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

#示例
df1 = make_df('ABC', range(3))
df2 = make_df('ABC', range(3,6))
pd.concat([df1, df2]) #按行合并
pd.concat([df1, df2],axis=1) #按列合并

'索引重复问题'
pd.concat([df1, df1])
#1 捕捉索引重复错误
pd.concat([df1, df1], verify_integrity=True)
#2 忽略索引
pd.concat([df1, df1], ignore_index=True)
#3 增加多级索引
pd.concat([df1, df1], keys=['x','y'])

'6.2 类似join的合并'
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
pd.concat([df5, df6], join='outer') #默认，并集连接
pd.concat([df5, df6], join='inner') #交集连接
pd.concat([df5, df6], join_axes=[df5.columns]) #join_axes指定结果列名



'7 内存式数据合并与连接'
'7.1 关系代数:pd.merge()'
#默认将共同列作为键进行合并
#一对一连接
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(df1)
print(df2)
df3 = pd.merge(df1, df2)
print(df3)

#多对一连接（有一列的值有重复）
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
pd.merge(df3, df4)

#多对多连接
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
pd.merge(df1, df5)


'7.2 设置数据合并的键'
#on合并相同列名
pd.merge(df1, df2, on='employee')
#合并不同列名
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)

#对索引的合并
df1a = df1.set_index('employee') #将employee列设为索引
df2a = df2.set_index('employee')
pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)

#索引和列混合使用：【left_index】 with 【right_on】 or left_on with right_index
pd.merge(df1a, df3, left_index=True, right_on='name')


'7.3 集合操作规则'
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])

pd.merge(df6, df7) #默认how='inner':交集
pd.merge(df6, df7, how='outer') #外连接，并集
pd.merge(df6, df7, how='left')  #左连接
pd.merge(df6, df7, how='right') #右连接


'7.4 列名重复 suffixes='
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})

pd.merge(df8, df9, on="name") #自动加后缀
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]) #手动加后缀

'7.5 案例：计算美国各州人口密度排名'
pop = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv')
areas = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv')
abbrevs = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv')

'合并州名与州缩写:pop与abbrevs'
merged = pd.merge(pop,abbrevs,how='outer',
         left_on="state/region", right_on="abbreviation").drop('abbreviation', axis=1)
merged.isnull().sum() #population 20 state  96
#检查缺失值
merged[merged['population'].isnull()].head()
#查看哪些state缺失了
merged.loc[merged['state'].isnull(), 'state/region'].unique()
#根据简称state/region，填充state
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
merged.tail()

'合并州人口与州面积:areas与merged'
final = pd.merge(merged,areas,how='outer',on='state')
final.isnull().sum() #population       20 area (sq. mi)    48
#查看哪些state的area缺失了
final.loc[final['area (sq. mi)'].isnull(), 'state'].unique()
#'United States' 全国面积缺失，删除该缺失值
final.dropna(inplace=True)

'计算2010年各州人口密度'
import numexpr
#2010年各州总人口，使用query查找
data2010 = final.query("year == 2010 & ages == 'total'") 
data2010.head()
#计算人口密度
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
density.sort_values(ascending=False,inplace=True)



'''8 累计与分组'''
'8.1 累计函数'
planets = sns.load_dataset('planets') #行星数据
# method  number  orbital_period（轨道周期）   mass  distance  year
planets.shape #（1035，6）
planets.notnull().sum()
planets.dropna().describe()
planets.dropna().mad() #均值绝对偏差（所有单个观测值与算术平均值的偏差的绝对值的平均）

'8.2 Groupby'
'按列取值'
#通过不同发现方法发现的行星的轨道周期
planets.groupby('method')['orbital_period'].median() 
#不同年份发现的行星的距离
plt.plot(planets.groupby('year')['distance'].median())
#可视化
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(30, 10), dpi=1000)
plt.bar(list(dict.fromkeys(planets['method'])), #dict.fromkeys：去重
        planets.groupby('method')['orbital_period'].median(),
       color='steelblue', width=0.8) 
plt.xticks(rotation=90,size=30)
plt.yticks(size=30)

'按组迭代'
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))

'调用方法'
planets.groupby('method')['year'].describe()
planets.groupby('method')['year'].describe().unstack() #分组统计

'8.3 aggregate(), filter(), transform(), and apply()'
'累计'
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6),
                   'data2': np.random.randint(0, 10, 6)}, columns=['key', 'data','data2'])
print(df)
df.groupby('key').sum()
df.groupby('key').aggregate([min, np.median, max])
df.groupby('key').aggregate({'data': 'min','data2': 'max'})

'过滤'
def filter_func(x):
    return x['data2'].std() > 4 #设置筛选条件，返回一个布尔值
df.groupby('key').filter(filter_func)

'数据转换，例如，标准化'
df.groupby('key').transform(lambda x: x - x.mean())

'apply():在每个组上应用任意方法，通过定义函数'
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data'] /= x['data'].sum()
    return x
df.groupby('key').apply(norm_by_data2)

'设置分割的键'
L = [0, 1, 0, 1, 2, 0] #将列表作为分组键 第1 3 6行一组 第2 4 行一组 第5行一组
df.groupby(L).sum()

df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'} #将索引映射到分组键
df2.groupby(mapping).sum()

df2.groupby(str.lower).mean() #python函数作为分组键
#将分组索引用str.lower从大写变成小写

'多个键构成的列表'
df2.groupby([str.lower,mapping]).mean() #str.lower和mapping组成分组键列表

'8.4 行星数据应用'
'获取不同方法和不同年份发现的行星数量'
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's' #具体年份转化为世纪

#method和decade组成分组键列表
planets.groupby(['method',decade])['number'].sum().unstack().fillna(0)



'''9 Pivot Tables'''
'9.1 简单数据透视表'
planets.pivot_table('number',index='method',aggfunc='sum',
                    columns=decade).fillna(0)

'9.2 多级数据透视表'
distancecut = pd.cut(planets['distance'],[1,40,200,8500]) #pd.cut:自选分箱 pd.qcut 给定区间数
#各种方法、各种距离、各种年代的行星总数
planets.pivot_table('number',index=['method',distancecut],aggfunc='sum',
                    columns=decade).fillna(0)
'''DataFrame.pivot_table(data, values=None, index=None, columns=None,
                      aggfunc='mean', fill_value=None, margins=False,
                      dropna=True, margins_name='All')'''

'''9.3 案例：美国人生日'''
plt.style.use('seaborn-whitegrid')
births = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv')
#新增一列，表示每个年代出生人数
births['decade'] = 10*(births['year']//10) #可以把196x转化为1960，以此类推
#数据透视表
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
#画出每一年的出生人数
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('total births per year')

#消除births异常值：【sigma-clipping】，删除births数在置信区间外的样本
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1] #均值
sig = 0.74 * (quartiles[2] - quartiles[0]) #均值稳定性估计
#QUery筛选
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
#原来1990年代后的出生数只精确到月，现在删掉了这些异常值

#将day列由字符串设为整数
births['day'] = births['day'].astype(int)
#将年月日组合，创建日期索引:pd.to_datetime
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek #dayofweek：获取该日期属于周几

#一周内每天出生多少人，按decade划分
births.pivot_table('births', index='dayofweek',
                    columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')

#按月划分，在几十年里每月每天一共出生了多少人
births_by_date = births.pivot_table('births', 
                                    [births.index.month, births.index.day])
#虚构一个年份，组成日期索引，来显示
births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
#结果：节假日时，出生人数急剧下降
#分析：医院放假，自己在家生



'10 向量化字符串'
'10.1 方法列表'
'''len()	lower()	translate()	islower()
ljust()	upper()	startswith()	isupper()
rjust()	find()	endswith()	isnumeric()
center()	rfind()	isalnum()	isdecimal()
zfill()	index()	isalpha()	split()
strip()	rindex()	isdigit()	rsplit()
rstrip()	capitalize()	isspace()	partition()
lstrip()	swapcase()	istitle()	rpartition()'''
data = ['peter', 'Paul', 'MARY', 'gUIDO']
names = pd.Series(data)
names.str.capitalize()
names.str.startswith('P')

'10.2 正则表达式'
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
#提取元素前的连续字母
monte.str.extract('([A-Za-z]+)', expand=False)

'10.3 其他方法'
'''get() slice() slice replace() cat()
repeat() normalize() pad() wrap() join() get_dummies()'''

#get_dummies()：当数据有一列包含若干coded indicator时用
full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
full_monte['info'].str.get_dummies('|')


'10.4 案例：食谱数据库'
'目标：根据现有食材快速找到食谱'
#读取数据：报错：Trailing data
pd.read_json(r'E:\2022Study\Data\Data Science book\recipeitems-latest.json\recipeitems-latest.json')

#将所有行JSON对象连起来
with open(r'E:\2022Study\Data\Data Science book\recipeitems-latest.json\recipeitems-latest.json', 'rb') as f:
    # Extract each line
    data = (line.strip() for line in f)
    # Reformat so each line is the element of a list
    data_json = "[{0}]".format(b','.join(data).decode())

#再次读取数据
recipes = pd.read_json(data_json)
recipes.shape  #(173278, 17)
recipes.iloc[0]

#查看食谱（ingredients）列表，统计列表的字符数分布
recipes.ingredients.str.len().describe()

#查看最长的食谱对应的菜：np.argmax 返回索引
np.argmax(recipes.ingredients.str.len()) #135598
recipes.name[np.argmax(recipes.ingredients.str.len())]

#查看哪些食谱是早餐：查看description中【含有】'breakfast'的
#[Bb]：允许大小写都可以匹配
recipes.description.str.contains('[Bb]reakfast').sum() #3524

#设计简易美食推荐系统
#提供简单食材列表（香料、调味料）
spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
#判断这些食材是否出现在recipes.ingredients中
import re
spice_df = pd.DataFrame(dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE))
                             for spice in spice_list))
spice_df.head()

#寻找使用'salt', 'pepper', 'oregano','cumin'的食材的ingredient
#使用query(),得到这些ingredient
selection = spice_df.query('salt&pepper&oregano&cumin')
print(selection)
#查看这些ingredient对应哪些菜
recipes.name[selection.index]



'''11 时间序列'''
'''11.1 Python日期与时间工具'''
'原生python：datetime和dateutil'
'解决时区:pytz'
#创建日期
from datetime import datetime
datetime(year=2022, month=11, day=23)
#解析字符串格式日期
from dateutil import parser
date = parser.parse("4th of July, 2015")

#打印星期
date.strftime('%A')

'Numpy：datetime64'
#将日期编码为64位整数
date = np.array('2022-07-04', dtype=np.datetime64)
date+np.arange(12) #之后的12天
#设置基本时间单位为纳秒
np.datetime64('2015-07-04 12:59:59.50', 'ns')

'Pandas'
#.to_datetime
date = pd.to_datetime("4th of July, 2015")
date.strftime('%A')


'11.2 创建时间索引:pd.DatetimeIndex'
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)

data['2014-07-04':'2015-07-04'] #切片
data['2015']


'11.3 Pd时间序列数据结构'
#对pd.to_datetime传递一个时间序列
dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
  '2015-Jul-6', '07-07-2015', '20150708'])
#.to_period('D')转化成peridindex类型
dates.to_period('D')
#时间差
dates - dates[0]

'有规律的时间序列'
pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=8) #按日变化
pd.date_range('2015-07-03', periods=8, freq='H') #按小时变化
pd.period_range('2015-07', periods=8, freq='M')
pd.timedelta_range(0, periods=10, freq="2H30T")

#BDay:工作日偏移序列
from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())


'11.4 重新取样、迁移和窗口（Resampling, Shifting, and Windowing)'
#导入google股票价格
from pandas_datareader import data
import datetime
symbol = 'AMZN'
data_source='google'
start_date = '2010-01-01'
end_date = '2016-01-01'
df = data.get_data_yahoo(symbol, start_date, end_date)



'''12 高性能运算：eval()与query()'''
'优点：省内存 查看内存：.values.nbytes'
'12.1 eval()'
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
                      for i in range(4))
pd.eval('df1 + df2 + df3 + df4') #优于df1 + df2 + df3 + df4
#.allclose；检查是否完全一致
np.allclose(df1 + df2 + df3 + df4,
            pd.eval('df1 + df2 + df3 + df4'))


'12.2 DataFrame.eval()：快速列间运算'
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
result1 = (df['A'] + df['B']) / (df['C'] - 1)
result3 = df.eval('(A + B) / (C - 1)')

#新增列
df.eval('D = (A + B) / C', inplace=True)

#使用局部变量：@表示这是一个变量名而不是列名
column_mean = df.mean(1)
result1 = df['A'] + column_mean
result2 = df.eval('A + @column_mean')


'12.3 DataFrame.query()'
#过滤运算
print(df.query('A < 0.5 and B < 0.5'))
#支持引用局部变量
Cmean = df['C'].mean()
print(df.query('A < @Cmean and B < @Cmean'))




























