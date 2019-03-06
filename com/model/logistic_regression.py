import statsmodels.api as sm
import pandas as pd

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
    woeData = pd.read_csv('WoeData.csv')
    logistic_regression(woeData)