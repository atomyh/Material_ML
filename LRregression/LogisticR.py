from numpy import *

# 加载数据
def load_data(input_data):
    Mat_fea = []
    Mat_lab = []
    file = open(input_data)
    for line in file.readlines():
        line_arr = line.strip().split()
        Mat_fea.append([1.0, float(line_arr[0]), float(line_arr[1])])
        Mat_lab.append(int(line_arr[2]))
    return Mat_fea, Mat_lab


# 计算sigmod的函数值
def sigmod(X):
    return 1.0 / (1 + math.exp(-X))


# predict_function
def prediction(X, Theta):
    res = sigmod(sum(X * Theta))
    if (res < 0.5):
        return 0
    else:
        return 1


# 初始化Theta
def iniTheta(m):
    return random.standard_normal([3, 1])


"""
Mat_fea:数据
Mat_label:标签
alpha:学习率
theta：参数权重
"""


def stocGradAscent0(Mat_fea, Mat_label, alpha):
    m, n = shape(Mat_fea)  # m*n = 100*3
    Theta = mat(iniTheta(n))  # 3*1
    for i in range(m):
        h = sigmod(sum(Mat_fea[i] * Theta))  #
        # print(h)
        erro = Mat_label[i] - h
        # print(shape(Mat_label[i]))
        # print(1 * Mat_fea[i])
        Theta = Theta + (alpha * erro * Mat_fea[i]).transpose()
        # print(Theta)
    return Theta


"""
随机选择学习率
随机选择样本训练
Iter表示迭代的轮数
"""


def stoGradAscent1(Mat_fea, Mat_label, Iter):
    m, n = shape(Mat_fea)
    Theta = mat(iniTheta(n))
    dataIndex = range(m)
    for i in range(Iter):
        for j in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            rand_inx = int(random.uniform(0, len(dataIndex)))
            #print(rand_inx)
            h=sigmod(sum(Mat_fea[rand_inx]*Theta))
            erro = Mat_label[rand_inx] - h
            Theta = Theta + (alpha * erro * Mat_fea[rand_inx]).transpose()/m
    return Theta

# print(h)


def LinearReg(fea, lab):
    m, n = shape(fea)  # m*n = 100*3
    right = 0
    Matrix_fea = mat(fea)
    Matrix__lab = mat(lab).transpose()

    # Theta = stocGradAscent0(Matrix_fea, Matrix__lab, 0.01)  # 3*1
    # for i in range(m):
    #     result = prediction(Matrix_fea[i], Theta)
    #     if result == Matrix__lab[i]:
    #         right += 1
    # accurcy = right / m
    # print(accurcy)  #accucy = 0.54
    ###################################################################################
    # Theta = stoGradAscent2(Matrix_fea, Matrix__lab, 150)
    # for i in range(m):
    #     result = prediction(Matrix_fea[i], Theta)
    #     if result == Matrix__lab[i]:
    #         right += 1
    # accurcy = right / m
    # print(accurcy)  #accurcy=0.95
    ####################################################################################

if __name__ == '__main__':
    Mat_fea, Mat_lab = load_data('testSet.txt')
    LinearReg(Mat_fea, Mat_lab)
