# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:17:01 2022

@author: Administrator
"""

'''1 二分查找'''
def binary_search(list,item):
    low = 0
    high = len(list) - 1
    
    while low <= high:
          mid = (low+high)/2
          guess = list[int(mid)]
          if guess == item:
             return int(mid)
          if guess < item:
             low = mid+1
          if guess > item:
             high = mid-1
    return None

my_list = [1,2,3,5,6,7]
print(binary_search(my_list,622))


'''4 快速排序'''
def quicksort(array):
    #basecase
    if len(array) <2: #数组包含一个/空元素时，认为是有序的
       return array
    #recursive case
    else:
        pivot = array[0] #选择基准值
        less = [i for i in array[1:] if i <= pivot] #小于基准值的子数组
        greater = [i for i in array[1:] if i > pivot] #大于基准值的子数组
        return quicksort(less) +[pivot] + quicksort(greater)

print(quicksort([10,5,6,2,8,4,3,9]))


'''6 广度优先'''
#表示图
graph = {}
graph['qzp'] = ['1','2','3']
graph['2'] = ['4']
graph['1'] = ['5']
graph['3'] = ['6','7']
graph['7'] = ['50','51','80']
graph['4'] = graph['5'] = graph['6'] = graph['50'] = graph['51'] = graph['80'] = []

#判断函数
def Person(number):
    return number[-1] == '6' #5开头的

from collections import deque 
def search(name):
    search_queue = deque() #空队列
    search_queue += graph[name] #qzp的邻居
    searched = [] #记录查找过的

    while search_queue: #队列不为空
          person = search_queue.popleft() #取出其中的第一个人
          if person not in searched: #这个没被找过：
             if Person(person): #满足需要找的条件
                print('这个人{}就是俺要找的人'.format(person))
             else: #不满足需要找的条件
                 search_queue += graph[person]# 把这个人的朋友都加到队列里，等待之后pop
                 searched.append(person) #把这个人标记为查过
search('qzp')


'''7 Dijkstra'''
processed = [] #处理过的节点

#权重表（散列表）
graph = {}
graph['start'] = {}
graph['start']['a'] = 6 #起点到a的权重为6
graph['start']['b'] = 2
graph['a'] = {}
graph['a']['fin'] = 1
graph['b'] = {}
graph['b']['a'] = 3
graph['b']['fin'] = 5
graph['fin'] = {}

#开销表
costs = {}
costs['a'] = 6
costs['b'] = 2
costs['fin'] = float('inf')

#父节点表
parents = {}
parents['a'] = 'start'
parents['b'] = 'start'
parents['fin'] = 'start'


#寻找最小开销
def find_lowest_cost_node(costs):
    lowest_cost = float('inf') #初始最低开销设为无穷大
    lowest_cost_node = None #初始，没找到最低开销节点
    for node in costs: #遍历所有节点
        cost = costs[node] #从开销表里取开销
        if cost < lowest_cost and node not in processed: #当前节点开销最低，且未被处理过
           lowest_cost = cost
           lowest_cost_node = node
    return lowest_cost_node
    
#算法实现
node = find_lowest_cost_node(costs) #当前最小开销node
while node is not None: #在所有node都被process后结束
      cost = costs[node]
      neighbors = graph[node] #寻找当前node的neighbor
      for n in neighbors.keys(): #遍历当前所有邻居
          new_cost = cost + neighbors[n]
          if new_cost < cost: #从起点经过该node前往neighbor比原来更近
             costs[n] = new_cost #更新该邻居的开销
             parents[n] = node #更新该邻居的父节点为node
      processed.append(node)
      node = find_lowest_cost_node(costs)
              
print(node)



'''8 贪婪算法'''
#近似算法
#州名
states_needed = set(['mt','wa','or','id',
                     'nv','ut','ca','az']) #传入数组，转换为集合

#广播台
stations = {}
stations['1'] = set(['id','nv','ut'])
stations['2'] = set(['wa','id','mt'])
stations['3'] = set(['or','nv','ca'])
stations['4'] = set(['nv','ut'])
stations['5'] = set(['ca','az'])
stations.items()

while states_needed: #不断循环，直到states_needed为空
      best_station = None
      states_covered = set()
      #station:键 states_for_station：station里的值
      for station, states_for_station in stations.items():
          covered = states_needed & states_for_station #当前广播台能覆盖而之前没覆盖的
          if len(covered) > len(states_covered): #当前广播台覆盖的比之前的best station还牛
             states_covered = covered
             best_station = station
          states_needed -= states_covered #更新需要覆盖的州
      print(best_station) #每次遍历station之后的best
print(best_station) #整个的best






