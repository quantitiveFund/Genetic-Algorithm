# -*- coding = utf-8 -*-
# @Time :  16:56
# @Author : cjj
# @File : funds_ga.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_50 = pd.read_excel('上证50.xlsx')
df_shidai = pd.read_excel('中欧时代先锋.xlsx')
df_500 = pd.read_excel('中证500.xlsx')
df_yiliao = pd.read_excel('医疗etf.xlsx')
df_bandaoti = pd.read_excel('半导体.xlsx')
df_fuguo = pd.read_excel('富国天惠.xlsx')
df_guangfa = pd.read_excel('广发双擎升级.xlsx')
df_yifangda = pd.read_excel('易方达.xlsx')
df_youse = pd.read_excel('有色金属.xlsx')
df_300 = pd.read_excel('沪深300.xlsx')
df_baijiu = pd.read_excel('白酒.xlsx')
df_alpha = pd.read_excel('东方阿尔法优势产业.xlsx')
df_jinxin = pd.read_excel('债券型金信民兴.xlsx')
df_xinnengyuan = pd.read_excel('创金新能源汽车.xlsx')
df_silu = pd.read_excel('博时丝路主题.xlsx')
df_dazong = pd.read_excel('大宗商品.xlsx')
df_qianyanyiliao = pd.read_excel('工银瑞信前沿医疗.xlsx')
df_guangfazhai = pd.read_excel('广发可转债.xlsx')
df_ditan = pd.read_excel('长信低碳环保.xlsx')
df_duoyinzi = pd.read_excel('长盛多因子策略.xlsx')

def pre_process(data):
    data.set_index(['日期'],drop = True,inplace = True)
    data = data.head(-2)
    data = data['收盘价(元)'].truncate(before = '2021-01-15')
    return data

data_list = [df_50,df_shidai,df_500,df_yiliao,df_bandaoti,df_fuguo,df_guangfa,df_yifangda,df_youse,df_300,
             df_baijiu,df_alpha,df_jinxin,df_xinnengyuan,df_silu,df_dazong,df_qianyanyiliao,df_guangfazhai,df_ditan,df_duoyinzi]
index_list = ["50","shidai","500","yiliao","bandaoti","fuguo","guangfa","yifangda","youse","300","baijiu",'alpha'
             ,'jinxin','xinenngyuan','silu','dazhong','qianyanyiliao','guangfazhai','ditan','duoyinzi']

df = pd.concat([pre_process(i) for i in data_list],axis = 1)
df.columns = index_list
# df.columns = ['close_50','close_shidai','close_500','close_yiliao','close_bandaiti','close_fuguo','close_guangfa','close_yifangda','close_youse','close_300','close_baijiu']
df.dropna(how='any',axis = 0,inplace= True)
df_yield = df.diff()/df.shift(1)
df_yield.dropna(how = 'any',inplace = True ,axis  = 0)
df_yield.reset_index(inplace = True,drop = True)
df.reset_index(inplace = True,drop = True)
df_yield = df_yield*100
yield_ = (df.tail(1).values - df.head(1).values)/df.head(1).values

#参数模块
assets_num = 20
pop_num = 100
min_share = 0.05

#创建一个100*n的矩阵
weights = np.random.random([pop_num,assets_num])


#weights 归一化
def weights_norm(weights):
    return weights / weights.sum(axis = 1).reshape(-1,1)

def get_portfolio(weights,df_yield,assets_num,pop_num):
    weights = weights_norm(weights)
    cov_mat = np.cov(df_yield.iloc[:,:assets_num].T)
    return_mat = np.dot(weights,yield_.reshape(-1,1))
    risk_mat = np.dot(np.dot(weights,cov_mat),weights.T)
    return weights,cov_mat,return_mat,risk_mat

weights,cov_mat,return_mat,risk_mat= get_portfolio(weights,df_yield,assets_num,pop_num)

#导入ga模块
POP_SIZE = len(weights)   #设定种群大小
CROSS_RATE = 0.8      #设定匹配可能性大小
MUTATION_RATE = 0.1    #基因变异的可能性大小
N_GENERATIONS = 300     #繁衍总次数
lst_best_sharp = []

#矩阵法
#适应性函数,计算夏普比率，无风险利率用0.05计算
def get_fitness(weights):
    weights,cov_mat,return_mat,risk_mat= get_portfolio(weights,df_yield,assets_num,pop_num)
    sharpratio = (return_mat-0.05)/ np.diag(np.sqrt(risk_mat)).reshape(-1,1)
    return sharpratio

def re_fitness(sharpratio):
    return (sharpratio -min(sharpratio) + 1e-3)*10

#选择留下来的人,
def select(weights, fitness):
    index = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                             p= np.exp(fitness.reshape(pop_num,))/(np.exp(fitness.reshape(pop_num,)).sum()))
    return weights[index]

#父母DNA结合过程
def crossover(parent, pop_copy):
    if np.random.rand() < CROSS_RATE:
        i_1 = np.random.randint(pop_num)
        i_ = np.random.randint(assets_num,size = assets_num//2)
        parent[i_] = pop_copy[i_1][i_]
    return parent

def update_mutate(iters):
    return MUTATION_RATE - 0.09*iters/N_GENERATIONS

#繁衍变异过程
def mutate(child,iters):
    if np.random.rand() < update_mutate(iters):
        child = np.random.random([1,assets_num])
    return child

def plotting():
    plt.cla()
    plt.xlim(0, 3)
    plt.ylim(-0.3, 1)
    plt.scatter(np.diag(np.sqrt(risk_mat)), return_mat, s=20, lw=0, c='red', alpha=0.1)
    plt.scatter(np.diag(np.sqrt(cov_mat)),yield_,s = 20, alpha = 0.6)
    plt.plot([0,np.sqrt(np.diag(risk_mat))[np.argmax(sharp_r)]],[0.05,return_mat[np.argmax(sharp_r)][0]],'y:')
    plt.xlabel('variance')
    plt.ylabel('expected return')
    plt.pause(0.05)

# 主循环部分
sharp_r = get_fitness(weights)
fitness = re_fitness(sharp_r)

for _ in range(N_GENERATIONS):


    # 遗传进化部分
    plotting()

    weights = select(weights, fitness)
    weights_copy = weights.copy()

    for i, parent in enumerate(weights):
        weights[i] = weights[i] / sum(weights[i])
        weights[i][weights[i] < min_share] = 0
        child = crossover(parent,weights_copy)
        child = mutate(child,_)
        weights[i] = child  # 下一代替换


    weights,cov_mat,return_mat,risk_mat= get_portfolio(weights,df_yield,assets_num,pop_num)
    sharp_r = get_fitness(weights)
    fitness = re_fitness(sharp_r)

    lst_best_sharp.append(sharp_r[np.argmax(sharp_r)])


print('完成循环')
print('Best_match:', weights[np.argmax(sharp_r)])
print('sharp_ratio:', sharp_r[np.argmax(sharp_r)])