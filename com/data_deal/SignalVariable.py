import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import math


# 变量分箱（binning）是对连续变量离散化（discretization）的一种称呼。信用评分卡开发中一般有常用的等距分段、等深分段、最优分段。
# 其中等距分段（Equval length intervals）是指分段的区间是一致的，比如年龄以十年作为一个分段；等深分段（Equal frequency intervals）是先确定分段数量，
# 然后令每个分段中数据数量大致相等；最优分段（Optimal Binning）又叫监督离散化（supervised discretizaion），
# 使用递归划分（Recursive Partitioning）将连续变量分为分段，背后是一种基于条件推断查找较佳分组的算法。


# 我们首先选择对连续变量进行最优分段，在连续变量的分布不满足最优分段的要求时，再考虑对连续变量进行等距分段。最优分箱的代码如下
# 定义自动分箱函数
def mono_bin(Y, X, n = 20):
    r = 0
    # 好人总数
    good=Y.sum()
    bad=Y.count()-good
    # abs 绝对值
    while np.abs(r) < 1:
        # pandas.qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise')
        # >>>x 要进行分组的数据，数据类型为一维数组，或Series对象
        #>>>q 组数，即要将数据分成几组，后边举例说明
        #>>>labels 可以理解为组标签，这里注意标签个数要和组数相等
        #>>>retbins 默认为False,当为False时，返回值是Categorical类型（具有value_counts()方法），为True是返回值是元组
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        # print(d1)
        d2 = d1.groupby('Bucket', as_index = True)
        # 斯皮尔曼相关性系数,计算当前这个特征与目标特征的相关性，，比皮尔逊相关系数的优点就是即便在变量值没有变化的情况下，
        # 也不会出现像皮尔森系数那样分母为0而无法计算的情况。另外，即使出现异常值，由于异常值的秩次通常不会有明显的变化
        # （比如过大或者过小，那要么排第一，要么排最后），所以对斯皮尔曼相关性系数的影响也非常小
        # 也就是说，我们不用管X和Y这两个变量具体的值到底差了多少，只需要算一下它们每个值所处的排列位置的差值，就可以求出相关性系数了
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)  # 计算 这个变量和目标特征的相关性，直到相关性的绝对值大于一
        # print(r,p)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    # 每个分组好人总数，因为好人是1 坏人是0
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    # 证据权重（Weight of Evidence,WOE）   每一个分组里面好人坏人数量之比除以总的好坏数量之比然后取对数 ，是一个单调递增函数
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    # 好人在每个分组的分布
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    # print(d3['min'], d3['max'], d3['sum'], d3['total'], d3['rate'])
    # print(d3['woe'], d3['goodattribute'], d3['badattribute'])
    d4 = (d3.sort_index(by = 'min'))
    print("=" * 60,'分箱情况')
    print(d4)
    cut=[]
    cut.append(float('-inf'))
    for i in range(1,n+1):
        qua=X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    woe=list(d4['woe'].round(3))
    return d4,iv,cut,woe

#自定义分箱函数
def self_bin(Y,X,cat):
    good=Y.sum()
    bad=Y.count()-good
    d1=pd.DataFrame({'X':X,'Y':Y,'Bucket':pd.cut(X,cat)})
    d2=d1.groupby('Bucket', as_index = True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_index(by='min'))
    print("=" * 60)
    print(d4)
    woe = list(d4['woe'].round(3))
    return d4, iv,woe

#证据权重（Weight of Evidence,WOE）转换可以将Logistic回归模型转变为标准评分卡格式。引入WOE转换的目的并不是为了提高模型质量，
# 只是一些变量不应该被纳入模型，这或者是因为它们不能增加模型值，或者是因为与其模型相关系数有关的误差较大，
# 其实建立标准信用评分卡也可以不采用WOE转换。这种情况下，Logistic回归模型需要处理更大数量的自变量。
# 尽管这样会增加建模程序的复杂性，但最终得到的评分卡都是一样的。
def replace_woe(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        value=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list

#计算分数函数
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores

#根据变量计算分数
def compute_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list

# 特征相关性分析
# 我们会用经过清洗后的数据看一下变量间的相关性。注意，这里的相关性分析只是初步的检查，进一步检查模型的VI（证据权重）作为变量筛选的依据
def feature_corr_analysis(data):
    corr = data.corr()  # 计算各变量的相关性系数
    xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']  # x轴标签
    yticks = list(corr.index)  # y轴标签
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})  # 绘制相关性系数热力图
    ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
    plt.show()

# 接下来，我进一步计算每个变量的Infomation Value（IV）。IV指标是一般用来确定自变量的预测能力
# IV=sum((goodattribute-badattribute)*woe)
# 通过IV值判断变量预测能力的标准是：
# < 0.02: unpredictive
# 0.02 to 0.1: weak
# 0.1 to 0.3: medium
# 0.3 to 0.5: strong
# > 0.5: suspicious
def feature_VI_analysis(ivlist,index):
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(index)) + 1
    ax1.bar(x, ivlist, width=0.4)  # 生成柱状图
    ax1.set_xticks(x)
    ax1.set_xticklabels(index, rotation=0, fontsize=12)
    ax1.set_ylabel('IV(Information Value)', fontsize=14)
    # 在柱状图上添加数字标签
    for a, b in zip(x, ivlist):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    plt.show()

# 利用逻辑回归模型进行训练
def logistic_regression(data):
    # 应变量
    Y = data['SeriousDlqin2yrs']
    # 自变量，剔除对因变量影响不明显的变量
    X = data.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                   'NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
    X1 = sm.add_constant(X)
    logit = sm.Logit(Y, X1)
    result = logit.fit()
    print(result.summary())

if __name__ == '__main__':

    # 6 变量选择 特征变量选择(排序)对于数据分析、机器学习从业者来说非常重要  附 Scikit-learn常用特征选取方法  https://www.cnblogs.com/hhh5460/p/5186226.html
    # 我们采用信用评分模型的变量选择方法，通过WOE分析方法，即是通过比较指标分箱和对应分箱的违约概率来确定指标是否符合经济意义
    data = pd.read_csv('TrainData.csv')
    pinf = float('inf')#正无穷大
    ninf = float('-inf')#负无穷大

    # 连续变量进行最优分段的分箱
    # SeriousDlqin2yrs--标签  RevolvingUtilizationOfUnsecuredLines--无担保放款的循环利用
    dfx1, ivx1,cutx1,woex1=mono_bin(data.SeriousDlqin2yrs,data.RevolvingUtilizationOfUnsecuredLines,n=10)
    dfx2, ivx2,cutx2,woex2=mono_bin(data.SeriousDlqin2yrs, data.age, n=10)
    dfx4, ivx4,cutx4,woex4 =mono_bin(data.SeriousDlqin2yrs, data.DebtRatio, n=20)
    dfx5, ivx5,cutx5,woex5 =mono_bin(data.SeriousDlqin2yrs, data.MonthlyIncome, n=10)

    # 连续变量离散化
    cutx3 = [ninf, 0, 1, 3, 5, pinf]  # NumberOfTime30-59DaysPastDueNotWorse
    cutx6 = [ninf, 1, 2, 3, 5, pinf]  # NumberOfOpenCreditLinesAndLoans
    cutx7 = [ninf, 0, 1, 3, 5, pinf]  # NumberOfTimes90DaysLate
    cutx8 = [ninf, 0,1,2, 3, pinf]    # NumberRealEstateLoansOrLines
    cutx9 = [ninf, 0, 1, 3, pinf]     # NumberOfTime60-89DaysPastDueNotWorse
    cutx10 = [ninf, 0, 1, 2, 3, 5, pinf] # NumberOfDependents
    # 使用自定义分箱
    dfx3, ivx3,woex3 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
    dfx6, ivx6 ,woex6= self_bin(data.SeriousDlqin2yrs, data['NumberOfOpenCreditLinesAndLoans'], cutx6)
    dfx7, ivx7,woex7 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTimes90DaysLate'], cutx7)
    dfx8, ivx8,woex8 = self_bin(data.SeriousDlqin2yrs, data['NumberRealEstateLoansOrLines'], cutx8)
    dfx9, ivx9,woex9 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)
    dfx10, ivx10,woex10 = self_bin(data.SeriousDlqin2yrs, data['NumberOfDependents'], cutx10)


    ivlist=[ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9,ivx10]
    index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
    # 通过VI值分析 来选取变量
    feature_VI_analysis(ivlist,index)


    # # 替换成woe
    data['RevolvingUtilizationOfUnsecuredLines'] = Series(replace_woe(data['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
    data['age'] = Series(replace_woe(data['age'], cutx2, woex2))
    data['NumberOfTime30-59DaysPastDueNotWorse'] = Series(replace_woe(data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    data['DebtRatio'] = Series(replace_woe(data['DebtRatio'], cutx4, woex4))
    data['MonthlyIncome'] = Series(replace_woe(data['MonthlyIncome'], cutx5, woex5))
    data['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(data['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
    data['NumberOfTimes90DaysLate'] = Series(replace_woe(data['NumberOfTimes90DaysLate'], cutx7, woex7))
    data['NumberRealEstateLoansOrLines'] = Series(replace_woe(data['NumberRealEstateLoansOrLines'], cutx8, woex8))
    data['NumberOfTime60-89DaysPastDueNotWorse'] = Series(replace_woe(data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
    data['NumberOfDependents'] = Series(replace_woe(data['NumberOfDependents'], cutx10, woex10))
    # data.to_csv('WoeData.csv', index=False)

    woeData = pd.read_csv('WoeData.csv')
    # 7 训练模型，算出回归系数
    # logistic_regression(woeData)

    test= pd.read_csv('TestData.csv')
    # # 替换成woe
    test['RevolvingUtilizationOfUnsecuredLines'] = Series(replace_woe(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
    test['age'] = Series(replace_woe(test['age'], cutx2, woex2))
    test['NumberOfTime30-59DaysPastDueNotWorse'] = Series(replace_woe(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    test['DebtRatio'] = Series(replace_woe(test['DebtRatio'], cutx4, woex4))
    test['MonthlyIncome'] = Series(replace_woe(test['MonthlyIncome'], cutx5, woex5))
    test['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(test['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
    test['NumberOfTimes90DaysLate'] = Series(replace_woe(test['NumberOfTimes90DaysLate'], cutx7, woex7))
    test['NumberRealEstateLoansOrLines'] = Series(replace_woe(test['NumberRealEstateLoansOrLines'], cutx8, woex8))
    test['NumberOfTime60-89DaysPastDueNotWorse'] = Series(replace_woe(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
    test['NumberOfDependents'] = Series(replace_woe(test['NumberOfDependents'], cutx10, woex10))
    test.to_csv('TestWoeData.csv', index=False)

    # 8 模型检验
    # logistic_regression(data)

    # 9 信用评分 我们已经基本完成了建模相关的工作，并用ROC曲线验证了模型的预测能力。接下来的步骤，就是将Logistic模型转换为标准评分卡的形式
    # 计算分数
    # #coe为逻辑回归模型的系数
    coe=[9.738849,0.638002,0.505995,1.032246,1.790041,1.131956]
    # # 我们取600分为基础分值，PDO为20（每高20分好坏比翻一倍），好坏比取20。
    p = 20 / math.log(2)  # 28.85390081777927
    q = 600 - 20 * math.log(20) / math.log(2)
    # 基础分 795
    baseScore = round(q + p * coe[0], 0)
    # # 各项部分分数  coe*w*factor 就是 权重*自变量 就是 Wt*X
    x1 = get_score(coe[1], woex1, p)
    x2 = get_score(coe[2], woex2, p)
    x3 = get_score(coe[3], woex3, p)
    x7 = get_score(coe[4], woex7, p)
    x9 = get_score(coe[5], woex9, p)
    print(x1,x2, x3, x7, x9)
    test1 = pd.read_csv('TestData.csv')
    test1['BaseScore']=Series(np.zeros(len(test1)))+baseScore
    test1['x1'] = Series(compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1))
    test1['x2'] = Series(compute_score(test1['age'], cutx2, x2))
    test1['x3'] = Series(compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3))
    test1['x7'] = Series(compute_score(test1['NumberOfTimes90DaysLate'], cutx7, x7))
    test1['x9'] = Series(compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9))

    # 个人总评分=基础分+各部分得分   就是 这个人的所有变量
    test1['Score'] = test1['x1'] + test1['x2'] + test1['x3'] + test1['x7'] +test1['x9']  + baseScore
    test1.to_csv('ScoreData.csv', index=False)
    # plt.show()