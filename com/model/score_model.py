import math

# 信用评分 我们已经基本完成了建模相关的工作，并用ROC曲线验证了模型的预测能力。接下来的步骤，就是将Logistic模型转换为标准评分卡的形式

def score_handler():
    # 计算分数
    # #coe为逻辑回归模型的系数
    coe = [9.738849, 0.638002, 0.505995, 1.032246, 1.790041, 1.131956]
    # # 我们取600分为基础分值，PDO为20（每高20分好坏比翻一倍），好坏比取20。
    p = 20 / math.log(2)
    q = 600 - 20 * math.log(20) / math.log(2)
    baseScore = round(q + p * coe[0], 0)
    print(p, q,coe[0],baseScore)
    # # 各项部分分数
    # x1 = get_score(coe[1], woex1, p)
    # x2 = get_score(coe[2], woex2, p)
    # x3 = get_score(coe[3], woex3, p)
    # x7 = get_score(coe[4], woex7, p)
    # x9 = get_score(coe[5], woex9, p)
    # print(x1, x2, x3, x7, x9)

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

#计算分数函数
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores

if __name__ == '__main__':
    score_handler()