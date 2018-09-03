from math import log
import operator

"""
1、计算香浓熵
2、切分子结构，计算各个特征的熵
3、选取最有子结构的
4、生成树

"""


# 测试用数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 计算香浓熵
def calShannonEnt(dataSet):
    size = len(dataSet)
    label_fea = {}
    for data_line in dataSet:  # 可以一次读一行数据
        k = data_line[-1]
        if k not in label_fea.keys():  # python字典结构的用法
            label_fea[k] = 0
        label_fea[k] += 1
    entro = 0.0
    for key in label_fea:
        prob = float(label_fea[key]) / size
        entro = entro - prob * log(prob, 2)
    return entro


# 切分数据集
def split_data_fea(dataSet, axis, value):
    retDataSet = []
    for feature_v in dataSet:
        if feature_v[axis] == value:
            retDataSet.append(feature_v)
    return retDataSet


"""
# 计算最大信息增益
S1:将每一个特征的类别提取出来
S2：根据特征类别来统计数值
S3：根据各个类别的数量计算概率
S4：根据最终的子表计算信息熵
以上完成一个特征熵的计算
"""


def calBestEntroy(dataSet):
    bestEntroy = 0.0
    tempEntroy = 0.0
    bestFeature = -1
    base_entroy = calShannonEnt(dataSet)
    feature_num = len(dataSet[0]) - 1
    # dataSet_num = len(dataSet)

    for fea in range(feature_num):  # 逐个特征计算
        list_fea = [data_line[fea] for data_line in dataSet]  # 按每个特征收集信息
        unique_fea = set(list_fea)  # 通过Set容器去掉重复值，得到这一个特征中所含有的类别
        # 统计特征中各个子类的数目
        for type in unique_fea:
            subSet = split_data_fea(dataSet, fea, type)
            prob = float(len(subSet)) / len(dataSet)
            # 计算当前特征类别子集的香农熵
            fea_entroy = calShannonEnt(subSet)
            tempEntroy += prob * fea_entroy
        InfoEntroy = base_entroy - tempEntroy

        if bestEntroy < InfoEntroy:  # 特征的选取通过按照信息增益从小到大排列
            bestEntroy = InfoEntroy
            bestFeature = fea
    return bestFeature  # 返回最优特征的下标值


# 获取出现最多的标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 建树！
"""
改进的算法从这里完善就好啦

"""


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:  # 好像是明白了机器学习实战中split_data_fea中为啥要费劲把特征丢掉了，就是这里啊以一个丢一个方便递归停止的判断啊。
        return majorityCnt(classList)
    bestFeat = calBestEntroy(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(split_data_fea(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataset, datalabel = createDataSet()
    # en = calShannonEnt(dataset)
    # en = split_data_fea(dataset, 0, 1)
    # en = calBestEntroy(dataset)
    tree = createTree(dataset, datalabel)
    print(tree)
