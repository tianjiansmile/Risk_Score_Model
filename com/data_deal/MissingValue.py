import pandas as pd
import matplotlib.pyplot as plt #导入图像库
from sklearn.ensemble import RandomForestRegressor

# 本脚本主要用户，数据的预处理，清洗


# 用随机森林对缺失值预测填充函数
# 在信用风险评级模型开发的第一步我们就要进行缺失值处理。缺失值处理的方法，包括如下几种。
# （1） 直接删除含有缺失值的样本。
# （2） 根据样本之间的相似性填补缺失值。
# （3） 根据变量之间的相关关系填补缺失值。
# 变量MonthlyIncome缺失率比较大，所以我们根据变量之间的相关关系填补缺失值，我们采用随机森林法
def set_missing(df):
    # 把已有的数值型特征取出来
    process_df = df.ix[:,[5,0,1,2,3,4,6,7,8,9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].as_matrix()
    unknown = process_df[process_df.MonthlyIncome.isnull()].as_matrix()
    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200,max_depth=3,n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print(predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df

# 数据预处理  在对数据处理之前，需要对数据的缺失值和异常值情况进行了解。Python内有describe()函数，可以了解数据集的缺失值、均值和中位数等
def data_init(data):
    data.describe().to_csv('DataDescribe.csv')  # 了解数据集的分布情况
    data = set_missing(data)  # 用随机森林填补比较多的缺失值
    data = data.dropna()  # 删除比较少的缺失值
    data = data.drop_duplicates()  # 删除重复项
    data.to_csv('MissingData.csv', index=False)
    data.describe().to_csv('MissingDataDescribe.csv')

#  缺失值处理
def data_handle(data):
    # 年龄等于0的异常值进行剔除
    data = data[data['age'] > 0]
    # 箱形图
    data379 = data[
        ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']]
    data379.boxplot()
    # 剔除异常值
    # 对于变量NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberOfTime60-89DaysPastDueNotWorse这三个变量，
    # 由下面的箱线图图3-2可以看出，均存在异常值，且由unique函数可以得知均存在96、98两个异常值，因此予以剔除。同时会发现剔除其中一个变量的96、98值，其他变量的96、98两个值也会相应被剔除。
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    data379 = data[
        ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']]
    # data379.boxplot()
    # plt.show()

# 数据探索性分析
# 在建立模型之前，我们一般会对现有的数据进行 探索性数据分析（Exploratory Data Analysis） 。
# EDA是指对已有的数据(特别是调查或观察得来的原始数据)在尽量少的先验假定下进行探索。常用的探索性数据分析方法有：直方图、散点图和箱线图等。
# 客户年龄分布如图4-1所示，可以看到年龄变量大致呈正态分布，符合统计分析的假设
def data_analysis(data):
    pass
# 数据切分  为了验证模型的拟合效果，我们需要对数据集进行切分，分成训练集和测试集
def data_split(data):
    from sklearn.model_selection import train_test_split
    Y = data['SeriousDlqin2yrs']
    X = data.ix[:, 1:]
    # 测试集占比30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # print(Y_train)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
    train.to_csv('TrainData.csv', index=False)
    test.to_csv('TestData.csv', index=False)

if __name__ == '__main__':
    #载入数据  1 缺失值处理
    data = pd.read_csv('cs-training.csv')
    data_init(data)

    # 2 异常值处理
    data_handle(data)

    # 目标变量SeriousDlqin2yrs取反
    data['SeriousDlqin2yrs'] = 1 - data['SeriousDlqin2yrs']

    # 4 数据探索性分析
    from com.data_deal import SignalVariable as sv
    # 查看各个特征以及目标特征之间的相关性
    sv.feature_corr_analysis(data)
    data_analysis(data)

    # 5 数据切分  为了验证模型的拟合效果，我们需要对数据集进行切分，分成训练集和测试集
    # data_split(data)




