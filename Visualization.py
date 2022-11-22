# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris

'''2 散点图'''
'plt.plot画'
plt.style.available[:10] #查看绘图风格
plt.style.use('seaborn-whitegrid')

x = np.linspace(0,10,30) #0到10共30个点
y = np.sin(x)

plt.plot(x, y, '-ok', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2) #'-ok':线条+代码+黑色
plt.ylim(-1.2, 1.2)

'plt.scatter画'
plt.scatter(x, y, marker='o')

rng = np.random.RandomState(0)
x = rng.randn(100) #服从标准正态
y = rng.randn(100) 
colors = rng.rand(100) #（0，1）均匀分布
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='coolwarm') #alpha：透明度
plt.colorbar()  # 右边显示颜色条 color scale

#iris dataset：三种属性的分类
iris = load_iris()
features = iris.data.T #四种属性，petal和sepal的长度和宽度

plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
#颜色：种类 x:sepal length y:width 散点大小:petal width
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

'效率：大数据集上，plt.plot性能更好'



'''3 可视化异常处理'''
'''3.1 基本误差线 errorbar'''
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='o',
             ecolor='lightgray',elinewidth=3,capsize=4) #fmt：控制线条和点外观


'''3.2 连续变量的误差'''
#高斯过程回归：对于带有不确定性的连续测量值进行拟合
from sklearn.gaussian_process import GaussianProcessRegressor

model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# 计算高斯过程拟合结果 Compute the Gaussian process fit
gp = GaussianProcessRegressor(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                     random_start=100)
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
dyfit = 2 * np.sqrt(MSE) #每个数据点的误差区间：2*sigma~95%置信区间

'''报错：cannot import name 'GaussianProcess' from 'sklearn.gaussian_process
改GaussianProcess为GaussianProcessRegressor后，报错 __init__() got an unexpected keyword argument 
'''

#结果可视化
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
#在plt.fill_between中给误差区间设置颜色
plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2) 
plt.xlim(0, 10);



'''4 三维数据可视化'''
'4.1 等高线图:plt.contour'
def f(x,y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y) #构建二维网格数据
Z = f(X, Y)
Z.shape #(40, 50)

#plt.contour等高线图
plt.contour(X, Y, Z, colors='black') #虚线代表负数
plt.contour(X, Y, Z, 20,cmap='YlGnBu') #等分为20  显示颜色变化
#plt.cm.+tab 键查看所有配色

#plt.contourf 填充等高线图
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar() #显示颜色条

#plt.imshow 将二维数组渲染成渐变图
plt.imshow(Z, extent=[0, 50, 0, 50], origin='lower',
           cmap='RdGy') #origin：调整原点位置从左上角到左下角（常规），extent：坐标范围
plt.colorbar()
plt.axis(aspect='image')

#彩色图上加等高线
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8) #等高线图

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5) #彩色图
plt.colorbar()

'4.2 频数直方图'
plt.style.use('seaborn-whitegrid')
data = np.random.randn(1000)
plt.hist(data, bins=30, alpha=0.7,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')

#histtype='stepfilled'搭配alpha
x1 = np.random.normal(0, 0.8, 10000)
x2 = np.random.normal(-2, 1, 10000)
x3 = np.random.normal(50, 20, 10000)

kwargs = dict(histtype='stepfilled', alpha=0.5,bins=400)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)

#仅计算每个bin中的样本数：
counts, bin_edges = np.histogram(data, bins=5)
print(counts)

'4.3 二维频数直方图'
mean = [0, 0]
cov = [[1, 1], [1, 2]]
#多元高斯分布创建样本数据
x, y = np.random.multivariate_normal(mean, cov, 1000000).T
x.shape #10000,

plt.hist2d(x, y, bins=300, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')

#只计算结果
counts, xedges, yedges = np.histogram2d(x, y, bins=30)

#六边形区分
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')

#用KDE抹掉空间中离散的数据点，让图更平滑
from scipy.stats import gaussian_kde

# 拟合数组维度 [Ndim, Nsamples]
data = np.vstack([x, y]) #np.vstack：x,y有相同列数，按行堆叠
data.shape #(2, 1000000)
kde = gaussian_kde(data)

# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40) #x轴刻度范围
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# Plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")



'''5 配置图例'''
'5.1 基本图例'
plt.style.use('classic')
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend()

#取消外边框
ax.legend(loc='upper left', frameon=False)
#设置标签列数
ax.legend(frameon=False, loc='lower center', ncol=2)
#fancybox:圆角边框  framealpha:外边框透明度 shadow：增加阴影 borderpad：
ax.legend(fancybox=True, framealpha=0.8, shadow=True, borderpad=1)


'5.2 在图例中显示不同尺寸的点'
import pandas as pd
cities = pd.read_csv(r"E:\2022Study\Data\Data Science book\california_cities.txt"
              ,encoding = "utf-8")
cities.info()

# 提取数据 lat：经度 lon：维度
lat, lon = cities['latd'], cities['longd']
#人口，区域面积
population, area = cities['population_total'], cities['area_total_km2']

#散点图标识四种数据，但不带标签
plt.scatter(lon, lat, label=None,
            c=np.log10(population), cmap='YlGnBu',
            s=area, linewidth=0, alpha=0.5)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

#添加图例
# 创建一些带标签的空列表[], []，也就是图例没有点，但显示了标签
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, framealpha=0.8, 
           fancybox=True,labelspacing=0.5,title='City Area')
plt.title('California Cities: Area and Population')


'5.3 显示多个图例'
#一般来说，标准的legend接口只能在一张图上显示一个图例。
fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':'] #设置4条线的不同风格
x = np.linspace(0, 10, 1000)

#画4条线
for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                     styles[i], color='black')
ax.axis('equal')

#lines[:2] 前两条线放进第一个图例中 
ax.legend(lines[:2], ['line A', 'line B'],
          loc='upper right', frameon=False)

#用Legend创建一个新对象，并用ax.add_artist()创造第二个图例
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
             loc='lower right', frameon=False)
ax.add_artist(leg)



'''6 配置color bar'''
cmap_list1 = plt.colormaps() #cmp颜色列表
import matplotlib._color_data as mcd
mcd.CSS4_COLORS.keys() #color颜色列表

'''6.1 顺序配色方案：binary/viridis
互逆配色方案：RdBu/PuOr
定性配色方案：rainbow.jet'''
plt.style.use('classic')
x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I,cmap='RdBu')
plt.colorbar()

'6.2 设置颜色刻度上下限'
speckles = (np.random.random(I.shape) < 0.01) #设置1%噪点
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar() #不设置颜色刻度上下限

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1) #设置颜色刻度上下限

'6.3 离散型颜色条'
#plt.cm.get_cmap函数，设置颜色、需要的颜色区间数量
plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1)



'7 多子图'
plt.style.use('seaborn-white')
'7.1 plt.axes'
#MATLAB风格接口：plt.axes(left, bottom, width, height) 左下角为0 右上角为1
#画中画:inset
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])

x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

#面向对象接口：fig.add_axes
#面向对象接口中，画图函数变成了显式的Figure和Axes
fig = plt.figure()
#上子图，起点y坐标0.5
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[], ylim=(-1.2, 1.2))
#下子图，起点y坐标0.1
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(-1.2, 1.2))

x = np.linspace(0, 10)
ax1.plot(np.sin(x)) #上面坐标轴无刻度
ax2.plot(np.cos(x))

'7.2 plt.subplot:创建简易网格子图'
#创建2行3列子图：
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),
             fontsize=18, ha='center')
    
#plt.subplots_adjust()：调整子图间间隔
#fig.add_subplot(): 面向对象式创建子图
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4) #横向间隔 纵向间隔
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
           fontsize=18, ha='center')

'7.3 plt.subplot[s]：一行代码创建网格'
fig, ax = plt.subplots(2, 3) #每个子图都有自己的标签
fig, ax = plt.subplots(2, 3, 
                        sharex='col', sharey='row') #每个子图都共享标签

#for遍历两轮，为每个子图做记号
#通过ax[i, j].text 做记号
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),
                      fontsize=18, ha='center')

'7.4 plt.GridSpec：不规则的子图网格'
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2]);



'8 文字(textual cue)与注释(annotation)'
'8.1 案例：数据处理部分详见Pandas部分'
#美国每日出生人数
plt.style.use('seaborn-whitegrid')
births = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv')

quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

births['day'] = births['day'].astype(int)

births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

#对特殊点加注释：各种节假日
style = dict(size=10, color='gray')
#3950：注释位置 ha：水平对齐方式
ax.text('2012-1-1', 3950, "New Year's Day", **style) 
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)

# 让日期居中
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

'8.2 坐标变化方式'
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

ax.text(0.5, 0.5, ". Data: (0.5, 0.5)", transform=ax.transData) #以数据为基准
ax.text(0.5, 0.5, ". Axes: (0.5, 0.5)", transform=ax.transAxes) #点在坐标轴上的位置
ax.text(0.5, 0.5, ". Figure: (0.5, 0.5)", transform=fig.transFigure) #点在图形上的位置

'8.3 箭头：plt.annotate()'
fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
#xy：箭头指向点 xytext：文字位置  shrink：箭头长度
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(facecolor='black',arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

'9 自定义刻度'
'9.1 添加次要刻度'
plt.style.use('classic')
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10000, 0, 10000])
ax = plt.axes(xscale='log', yscale='log')
ax.grid()

print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())

print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())


'9.2 隐藏刻度与标签：plt.NullLocator() plt.NullFormatter()'
ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator()) 
ax.xaxis.set_major_formatter(plt.NullFormatter())


'9.3 增减刻度数量'
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
#为每个坐标轴设置刻度上限
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(2)) #每个子图x轴最多2个刻度
    axi.yaxis.set_major_locator(plt.MaxNLocator(4))
fig


'9.4 更改刻度值'
#正余弦曲线
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi) #此时x轴刻度是整数

#更改刻度
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2)) #设置大刻度值为Π/2
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4)) #设置小刻度值为Π/4
plt.show()
#自定义刻度设置标签：plt.FuncFormatter
def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.show()

'9.5 绘图风格'
plt.style.available[:10]



'''10 Matplotlib画三维图'''
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

'10.1 三维数据点和线'
#线
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'blue')

#点
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='RdGy')
plt.show()


'10.2 三维等高线'
#输入二维网格数据
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)

#由函数计算高度
def f(x, y,i):
    return np.sqrt(abs(x**i-y**i))

#contour3D
for i in range(1,20,2):
    Z = f(X, Y, i)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 500, cmap= plt.colormaps()[i])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(i*10,i*2) #ax.view_init:调整观察角度(平面角、俯仰角)
    plt.show()

#三维网线图：plot_wireframe
import matplotlib._color_data as mcd
mcd.CSS4_COLORS.keys() #color词典

for i in range(1,20,2):
    Z = f(X, Y, i)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, Z, color = list(mcd.CSS4_COLORS.keys())[i])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(i*10,i*2) #ax.view_init:调整观察角度(平面角、俯仰角)
    plt.show()

#三维曲面图:.plot_surface
for i in range(1,20,2):
    Z = f(X, Y, i)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                 cmap= plt.colormaps()[i],edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(i*10,i*2)
    ax.set_title('surface')
    plt.show()

#三角剖分曲面:ax.plot_trisurf
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
plt.figure(figsize=(40, 20),dpi = 2000)
for i in range(1,20,1):
    z = f(x, y,i)
   #散点图
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

    #由ax.plot_trisurf通过连接点形成三角形来创造曲面
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z,
                cmap=plt.colormaps()[i], edgecolor='none')
    ax.view_init(i*10,i*2)
    ax.set_title('Surface Triangulations')
    plt.show()



'''11 地理数据可视化：Basemap'''
from mpl_toolkits.basemap import Basemap
'画地球投影'
plt.figure(figsize=(30, 30),dpi=2000)
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
m.bluemarble(scale=0.5)

fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6, 
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5) #背景；etopo 陆地和海底的地形

#添加城市坐标标记
x, y = m(-122.3, 47.6)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Seattle', fontsize=12);



'''12 Seaborn'''
import seaborn as sns
sns.set()
'12.1 频数直方图、KDE和密度图'
#二元数据
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col],alpha=0.5) #x和y分别的直方图
for col in 'xy':
    sns.kdeplot(data[col], shade=True)

#distplot：频数直方图与KDE结合
sns.distplot(data['x'])
sns.distplot(data['y'])

#jointplot:二维kde的联合分布
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde',cmap='viridis') 
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex',cmap='YlGnBu') #六边形


'12.2 矩阵图 sns.pairplot'
#多维数据可视化
iris = sns.load_dataset("iris")
iris.head()
sns.pairplot(iris, hue='species', size=2.5)

'12.3 分面频次直方图 sns.FacetGrid'
#数据子集的频次直方图
tips = sns.load_dataset('tips')
tips.head()
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))

'12.4 因子图 .factorplot'
#分组箱线图 x轴：星期 y轴：小费 分组：性别
#sns.factorplot（‘星期’，‘小费’，‘性别’）
with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill")

'12.5 联合分布图 .jointplot'
with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex') #六边形联合分布
    sns.jointplot("total_bill", "tip", data=tips, kind='reg') #联合分布作回归

'12.7 条形图 .factorplot'
planets = sns.load_dataset('planets')
planets.head()
#简单条形图
with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)
#对比不同方法发现行星的数量，是因子图的特殊形式
with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2.0, kind='count',
                       hue='method', order=range(2001, 2015)) #aspect:长度设置
    g.set_ylabels('Number of Planets Discovered')
    
'12.6 小提琴图 .violinplot'
#性别
sns.violinplot('sex','tip_pct',
               data=tips,palette=['lightblue','lightpink'])
#性别+星期
with sns.axes_style(style=None):
    sns.violinplot("day", "tip_pct", hue="sex", data=tips,
                   split=True, inner="quartile",
                   palette=["lightblue", "lightpink"])



'''13 Scikit-plot'''
'https://github.com/reiinakano/scikit-plot'
import scikitplot as fuckmatplotlib #！
import scikitplot
from collections import Counter
from sklearn.datasets import load_digits #手写数字数据集
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
X.shape #1797*64 对应到一个8x8像素点组成的矩阵，每一个值是其灰度值
Counter(y) #从0-9
num = np.array(X[0]).reshape((8,8))
plt.matshow(num) #输出第一个数字像素

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
nb = GaussianNB()
nb.fit(X_train, y_train)
predicted_probas = nb.predict_proba(X_test)

'13.1 指标模块'
'9个数字分别的ROC曲线'
fuckmatplotlib.metrics.plot_roc(y_test, predicted_probas)
plt.show()

'PR曲线'
plt.colormaps()
fuckmatplotlib.metrics.plot_precision_recall_curve(y_test,
                 predicted_probas, cmap='RdGy_r')
plt.show()

'混淆矩阵'
from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(nb, X, y)
plot = fuckmatplotlib.metrics.plot_confusion_matrix(y, predictions, normalize=True)
plt.show()

'KS统计图'
#反映二分类模型的区分能力
#对一个风控模型来说，经验上，KS 统计量至少要 40% 才反应一个较好的判别能力
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

model = LogisticRegression()
model2 = GaussianNB()
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
probas = model.predict_proba(X_test)
scikitplot.metrics.plot_ks_statistic(y_true=y, y_probas=probas)
plt.savefig('pic/model_classification_ks.jpg') #保存图像


'13.2 估算器模块'
'learning curve'
nb = GaussianNB()
scikitplot.estimators.plot_learning_curve(nb, X_train, y_train)
plt.show()

'feature importance'
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
scikitplot.estimators.plot_feature_importances(rf,
                                               feature_names=['1','2'])


'13.3 聚类和分解模块'
'Kmeans聚类'
#scikitplot.cluster.plot_elbow_curve
from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=10)#构造聚类器
estimator.fit(X)
#elbow plot
scikitplot.cluster.plot_elbow_curve(estimator, X,cluster_ranges=range(1, 20))
cluster_labels = estimator.fit_predict(X)
#轮廓系数
scikitplot.metrics.plot_silhouette(X, cluster_labels) 

'PCA'
pca = PCA(random_state=1)
pca.fit(X)
#Explained variance
scikitplot.decomposition.plot_pca_component_variance(pca)
scikitplot.decomposition.plot_pca_2d_projection(pca, X, y)










