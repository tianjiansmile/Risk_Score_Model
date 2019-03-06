import pandas as pd
import matplotlib.pyplot as plt #导入图像库
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import missingno


def data_check(data):
    # print(data.head())

    # 数据的维度查看
    print(data.shape)

    # 一个简单的统计
    # data.describe().to_csv('Q4_Describe.csv')

    # 缺失值的可视化
    # missingno.matrix(data.sample(500),labels=True)

    # 有的特征空值较多，缺失比例0.5以上的特征全部删除
    data = data.dropna(thresh=len(data)*0.5,axis=1)

    data = data.drop_duplicates()  # 删除重复项

    # 检查空值比例
    check_null = data.isnull().sum(axis=0).sort_values(ascending=False) / float(len(data))
    print(check_null[check_null > 0])

    # data.to_csv('Q4_handled.csv', index=False)
    # data.describe().to_csv('Q4_handled_Describe.csv')


    # plt.show()

#  缺失值处理
def data_handle(data):

    pass

# 手动选取一些特征
def feature_remove(data):
    col = ['loan_status','loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
           'annual_inc', 'verification_status', 'issue_d', 'purpose', 'addr_state', 'dti', 'delinq_2yrs',
           'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
           'acc_now_delinq', 'pub_rec_bankruptcies', 'tax_liens']
    # 重置
    data = data[col]
    new_feature = {'loan_amnt':'贷款金额','term':'贷款期限','int_rate':'利率','installment':'每月还款金额'
                    ,'grade':'贷款等级','sub_grade':'基础等级','emp_length':'工作年限','home_ownership':'房屋所有权(出租 自有 按揭 其他)'
                    ,'annual_inc':'年收入','verification_status':'收入是否由LC验证','issue_d':'放款日期','loan_status':'是否借款'
                    ,'purpose':'贷款目的','addr_state':'所在地','dti':'月负债比','delinq_2yrs':'过去两年借款人逾期30天以上的次数',
                   'earliest_cr_line':'信用报告最早日期','inq_last_6mths':'过去6个月被查询次数'
                    ,'open_acc':'为还清的贷款额度','pub_rec':'摧毁公共记录的数量','revol_bal':'总贷款金额','revol_util':'额度循环使用率'
                    ,'total_acc':'总贷款笔数','acc_now_delinq':'拖欠的账户数量','pub_rec_bankruptcies':'公开记录破产的数量'
                    ,'tax_liens':'留置税数量'}
    #
    # data = data.rename(colnums=new_feature)

    # print(data.head())

    return data,col

def set_missing(df,emp_length_loc):
    # 把已有的数值型特征取出来
    process_df = df.ix[:, [7, 0, 1, 2, 3, 4, 6, 8, 9,10,11,12,13,
                           14,15,16,17,18,19,20,21,22,23,24,25]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.emp_length.notnull()].as_matrix()
    unknown = process_df[process_df.emp_length.isnull()].as_matrix()

    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]

    # print(X.shape, y.shape)
    # print(X, y)

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知特征值预测
    # predicted = rfr.predict(unknown[:,1:]).round(0)
    # print(predicted)

if __name__ == '__main__':
    data = pd.read_csv('LoanStats_2016Q4.csv',skiprows=1)

    # 手动筛选一些变量,如果不需要，跳过这一步
    data,col = feature_remove(data)

    # 1 数据预处理
    data_check(data)

    print(data.shape)
    # print(len(continuous_var))
    # emp_length_loc = col.index('emp_length')
    # revol_util_loc = col.index('revol_util')
    # dti_loc = col.index('dti')
    # print(emp_length_loc,revol_util_loc,dti_loc)
    # set_missing(data,emp_length_loc)
