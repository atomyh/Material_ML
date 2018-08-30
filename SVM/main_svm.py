from numpy import *


class SVM:
    def __init__(self, dataMatIn, dataMatLabels, C, toler, maxIter, parm_kernel):
        self.dataMatIn = dataMatIn
        self.dataMatLabels = dataMatLabels
        self.C = C
        self.toler = toler
        # 输入数据的行数
        self.volum = shape(dataMatIn)[0]
        self.b = 0
        # 初始化alpha变量
        self.alphas = mat(zeros((self.volum, 1)))
        # 最大迭代次数
        self.maxIter = maxIter
        # 默认核函数
        self.parm_kernel = ['linear_kernel', 0]
        self.eCache = mat(zeros((self.volum, 2)))
        self.K = mat(zeros((self.volum, self.volum)))
        for ind in range(self.volum):
            self.K[:, ind] = kernelTrans(self.dataMatIn, self.dataMatIn[ind, :], self.parm_kernel)


def load_data(file_name):
    """
    readline():一次读取一行数据，array
    readlines():一次读取整个文本，把数据做成行列表
    :param file_name:
    :return:
    """
    data_feature = []
    data_label = []
    file_obj = open(file_name)
    for line in file_obj.readlines():
        # print(type(line))
        line_list = line.strip().split()
        data_feature.append([float(line_list[0]), float(line_list[1])])
        data_label.append(float(line_list[2]))
    # print(data_feature)
    # print(data_label)
    return data_feature, data_label


"""
X:所有的数据矩阵
k：要计算的行
parameter[0]:核函数类型——————>linear_kernel=线性核函数；guass_kernel=高斯核函数
parameter[1]函数的参数
"""


def kernelTrans(X, A, parameter):
    m, n = shape(X)
    # param = diag(parameter[2],m)
    k_result = mat(zeros((m, 1)))
    if (parameter[0] == 'linear_kernel'):
        k_result = X * A.T
    elif (parameter[0] == 'guass_kernel'):  # 高斯核函数
        for j in range(m):
            delta_col = X[j] - A
            k_result[j] = delta_col * delta_col.T
        k_result = exp(k_result / (-2 * parameter[1] ** 2))
    #print("核函数计算完成")
    return k_result


# 计算损失
def calEi(svm_obj, index):
    fx_i = float(multiply(svm_obj.alphas, svm_obj.dataMatLabels).T * svm_obj.K[:, index] + svm_obj.b)
    E_i = fx_i - float(svm_obj.dataMatLabels[index])
    return E_i


# 随机取alpha
def pickJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


# 更新数据结构
def upEcache(k, svm_obj):
    Ek = calEi(svm_obj, k)
    svm_obj.eCache[k] = [1, Ek]


# 随机获取值
def pickJ(i, Ei, svm_obj):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    svm_obj.eCache[i] = [1, Ei]
    validaEcacheList = nonzero(svm_obj.eCache[:, 0].A)[0]  # 非零E值对应的alpha值
    if (len(validaEcacheList)) > 1:
        for k in validaEcacheList:
            if k == i:
                continue
            Ek = calEi(svm_obj, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK, Ej
    else:
        j = pickJrand(i, svm_obj.volum)
        Ej = calEi(svm_obj, j)
    return j, Ej


def clipAlpha(aj, H, L):  # 保证a在L和H范围内（L <= a <= H）
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def inLoop(i, svm_ob):
    Ei = calEi(svm_ob, i)
    # 这个地方可是搞死我了！
    # 这里为啥通过C就可以判断KKT条件在李航的书中已经介绍了，这里一定有一个思想先找支持向量的点，然后！在这些点找可优化滴！并且是一侧一侧的找！
    if ((svm_ob.dataMatLabels[i] * Ei < -svm_ob.toler) and (svm_ob.alphas[i] < svm_ob.C)) or (
            (svm_ob.dataMatLabels[i] * Ei > svm_ob.toler) and (svm_ob.alphas[i] > 0)):
        # 随机选取一个j计算相应的值
        alpha_j, Ej = pickJ(i, Ei, svm_ob)
        # 两个alpha值已经有了下面进入alpha值得更新
        # 开辟新空间存储旧的alpha值
        alphaiOdd = svm_ob.alphas[i].copy()
        alphajOdd = svm_ob.alphas[alpha_j].copy()
        # 更新alpha
        if svm_ob.dataMatLabels[i] != svm_ob.dataMatLabels[alpha_j]:
            L = max(0, svm_ob.alphas[alpha_j] - svm_ob.alphas[i])
            H = min(svm_ob.C, svm_ob.C + svm_ob.alphas[alpha_j] - svm_ob.alphas[i])

        else:
            # svm_ob.dataMatLabels[i] == svm_ob.dataMatLabels[alpha_j]:
            L = max(0, -svm_ob.C + svm_ob.alphas[alpha_j] + svm_ob.alphas[i])
            H = min(svm_ob.C, svm_ob.alphas[alpha_j] + svm_ob.alphas[i])
        if L == H:
            print("L与H相等")
            return 0

        eta = svm_ob.K[1, 1] + svm_ob.K[2, 2] - 2 * svm_ob.K[1, 2]

        if eta != 0:
            svm_ob.alphas[alpha_j] += (svm_ob.dataMatLabels[alpha_j] * (Ei - Ej)) / eta
            svm_ob.alphas[alpha_j] = clipAlpha(svm_ob.alphas[alpha_j], H, L)

        # 更新数据结构
        upEcache(alpha_j,svm_ob)
        # 观察alphaj下降的程度

        if (abs(svm_ob.alphas[alpha_j] - alphajOdd) > svm_ob.toler):
            print("alphj下降度不够多")
            return 0
            # 更新alphai
        svm_ob.alphas[i] += svm_ob.dataMatLabels[i] * svm_ob.dataMatLabels[alpha_j] * (
                    alphajOdd - svm_ob.alphas[alpha_j])
        upEcache(i,svm_ob )

        # 更新b值
        b1 = -Ei - svm_ob.dataMatLabels[i] * svm_ob.K[1, 1] * (svm_ob.alphas[i] - alphaiOdd) - svm_ob.dataMatLabels[
            alpha_j] * svm_ob.K[1, 2] * (svm_ob.alphas[alpha_j] - alphajOdd) + svm_ob.b
        b2 = -Ej - svm_ob.dataMatLabels[i] * svm_ob.K[1, 2] * (svm_ob.alphas[i] - alphaiOdd) - svm_ob.dataMatLabels[
            alpha_j] * svm_ob.K[2, 2] * (svm_ob.alphas[alpha_j] - alphajOdd) + svm_ob.b
        if (0 < svm_ob.alphas[i] < svm_ob.C):
            svm_ob.b = b1
        elif (0 < svm_ob.alphas[alpha_j] < svm_ob.C):
            svm_ob.b = b2
        else:
            svm_ob.b = (b1 + b2) / 2
        return 1
    else:
        return 0






"""
SMO：最快求解alpha和b
data_feature: 数据特征
data_label  :数据类别
tolar       :阈值
maxIter     :最大迭代次数
KTup        :核函数选择
"""


def SMO(data_feature, data_label, C, tolar, maxIter, param):
    svm_ob = SVM(mat(data_feature), mat(data_label).transpose(), C, tolar, maxIter, parm_kernel=param)
    # 遍历方法，迭代条件、alpha值是否改变
    numIter = 0
    isAlphaChange = 1  # alpha是否已经优化
    entireSet = True
    while (numIter < maxIter) and (isAlphaChange > 0):
        isAlphaChange = 0
        if(entireSet):
            for i in range(svm_ob.volum):
                isAlphaChange += inLoop(i, svm_ob)
                #print("full set ,iter:%d,i:%d,changed:%d" % numIter, i, isAlphaChange)
            numIter += 1
        else:
            nonBoundIs = nonzero((svm_ob.alphas.A > 0) * (svm_ob.alphas.A < C))[0]
            for i in nonBoundIs:
                isAlphaChange += inLoop(i, svm_ob)
                #print("full set ,iter:%d,i:%d,changed:%d" % numIter, i, isAlphaChange)
            numIter += 1
        if entireSet:
            entireSet = False
        elif (isAlphaChange == 0):
            entireSet = True
        #print("iteration number: %d" % iter)
    return svm_ob.b, svm_ob.alphas

# svm核心算法
def svm(train_data, test_data):
    train_feature, train_label = load_data(train_data)
    # 一会儿再商议最后使用什么核函数
    trMatfea = mat(train_feature)
    trMatlab = mat(train_label).transpose()

    b, alpha = SMO(train_feature, train_label, 250, 0.0001, 10000, param=['guass_kernel', 1.3])
    #print(shape(alpha))

    # 找出支持向量来————>支持向量不为零
    svIndex = nonzero(alpha)[0]
    svDatafea = trMatfea[svIndex] #(19,2)
    svDatalab = trMatlab[svIndex] #(19,1)
    # 支持向量与训练数据得到一个预测结果
    #print(svDatalab)
    #print(multiply(svDatalab, alpha[svIndex]))
    m,n = shape(trMatfea)
    for i in range(m):
        kernelEval = kernelTrans(svDatafea, trMatfea[i,:],['guass_kernel',1.5])
        #print(kernelEval)
        predict = kernelEval.T*multiply(svDatalab, alpha[svIndex])+b
        #print(kernelEval.transpose()*multiply(svDatalab, alpha[svIndex]))
        #print(shape(alpha[svIndex]))
        #print(b)

    # 让支持向量跟测试数据做预测得到一个结果


def main():
    train_data = 'train.txt'
    test_data = 'test.txt'
    # train_feafure, train_label = load_data(train_data)
    # test_feature, test_label = load_data(test_data)
    svm(train_data, test_data)


if __name__ == '__main__':
    main()
