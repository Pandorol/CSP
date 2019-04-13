import numpy as np


def mulity_csp(EEG_X, kind):
    """
    针对多类 CSP问题
    EEG_X分量分别是一个 n × T×channel 的矩阵数据   分量EEG_X = A*S
    A是一个满秩的未知 n × n 矩阵。
    S一个 n×T 数据矩阵(源信号)具有以下性质：
        a) 对于每个t, S(:，t)的分量在统计上是独立的；
        b) 对于每个p, S(p，:)是零均值“源信号”的实现。
    特征矩阵的联合近似对角化

    输入 :
            EEG_X1: 它们的每一列都是n个传感器的样本
            kind: 多类任务数目
    输出 :
            X_Fsp: 各自的过滤器
            EEG_S：分别是估计源信号
            d: 取出特征的数目
    """

    EEG_X = np.mat(np.array(EEG_X, dtype=np.complex128))
    num_channel, num_samples = EEG_X.shape      # num_channel代表通道，num_samples代表总样本数量
    num_samples = int(num_samples/kind)

    channel = range(num_channel)
    samples = range(num_samples)

    '''数据处理'''
    ## 零均值
    EEG_X_zeros = np.zeros((num_channel, num_samples*kind))
    for k in range(kind):
        for p in channel:
            for q in samples:
                EEG_X_zeros[p, q+k*num_samples] = EEG_X[p, q+k*num_samples] - np.mean(EEG_X[p, k*num_samples:(k+1)*num_samples])

    ## 白化处理
	EEG_X_cov = np.mat(np.zeros((num_channel, num_channel*kind)))
    # 计算各自的的协方差矩阵; 并计算混合协方差矩阵
	for k in range(kind):
		EEG_X_cov[:, k*num_channel:(k+1)*num_channel] = np.dot(EEG_X_zeros[:, k*num_samples:(k+1)*num_samples], np.mat(EEG_X_zeros[:, k*num_samples:(k+1)*num_samples]).H) \
            / np.trace(np.dot(EEG_X_zeros[:, k*num_samples:(k+1)*num_samples], np.mat(EEG_X_zeros[:, k*num_samples:(k+1)*num_samples]).H))

    EEG_X_mixcov = np.mat(np.zeros((num_channel, num_channel)))
    for k in range(kind):
        EEG_X_mixcov += EEG_X_cov[:, k*num_channel:(k+1)*num_channel]

    # 混合协方差矩阵特征分解
    vals, Vecs = np.linalg.eig(EEG_X_mixcov)
    Vals       = np.mat(np.diag(vals))

    # 得到白化矩阵
    white = np.dot(np.mat(np.sqrt(Vals)).I, np.mat(Vecs).H)

    EEG_X_white = np.mat(np.zeros((num_channel, num_channel*kind)))
    ## 对X的协方差矩阵进行变换
    for k in range(kind):
        EEG_X_white[:, k*num_channel:(k+1)*num_channel] = np.dot(np.dot(white, EEG_X_cov[:, k*num_channel:(k+1)*num_channel]), white.H)
    EEG_M_diag = EEG_X_white

    '''进行近似联合对角化处理'''
    Giv  = np.eye(num_channel)
    amin = 100  # 结束条件
    while amin > 25:
        amin = 0
        p = np.random.randint(0, num_channel-1)
        q = np.random.randint(p+1, num_channel)
        h = np.zeros((3, 1))
        for k in range(kind):
            h = h + np.mat(([EEG_M_diag[p, p+k*num_channel]-EEG_M_diag[q, q+k*num_channel]],[EEG_M_diag[p, q+k*num_channel]+EEG_M_diag[q, p+k*num_channel]],[np.complex(0,1)*(EEG_M_diag[q, p+k*num_channel]-EEG_M_diag[p, q+k*num_channel])]))

        # 取特征值，特征向量
        h_vals, h_Vecs = np.linalg.eig(np.dot(np.real(h),np.real(h.H)))

        h_sort_vals = sorted(h_vals)  # 从小到大排列
        h_Id        = np.argsort(h_vals)

        angles = h_Vecs[:, h_Id[-1]]
        # 防止出现无法计算的情况
        if angles[0] < 0:
            angles = -angles

        # 计算r,s,c的值
        r = np.sqrt(abs(angles[0]) ** 2 + abs(angles[1]) ** 2 + abs(angles[2]) ** 2)[0,0]
        c = np.sqrt((angles[0] + r) / (2 * r))[0,0]
        s = ((angles[1] - np.complex(0,1)*angles[2]) / np.sqrt(2 * r * (angles[0] + r)))[0,0]


        ## 通过Givens旋转更新矩阵M和V
        Givens       = np.eye(num_channel)
        Givens[p, p] = c
        Givens[p, q] = np.conj(s)
        Givens[q, p] = -s
        Givens[q, q] = np.conj(c)
        Giv          = np.dot(Giv, Givens)

        # 进行对角化
        for k in range(kind):
            EEG_M_diag[:, k*num_channel:(k+1)*num_channel] = np.dot(np.dot(np.mat(Givens).H, EEG_M_diag[:, k*num_channel:(k+1)*num_channel]), Givens)
            amin += np.sum(abs(EEG_M_diag[:, k*num_channel:(k+1)*num_channel])**2) - np.sum(abs(np.diag(EEG_M_diag[:, k*num_channel:(k+1)*num_channel]))**2)
        print(amin)

    '特征选取'

    ## 计算空间滤波器
    Fsp = np.dot(np.mat(Giv).H, white)

    I      = np.zeros((kind, num_channel))
    I_sort = np.zeros((kind,num_channel))
    I_Id   = np.zeros((kind,num_channel))
    # 计算I, 寻找最大的值
    for k in range(kind):
        for q in channel:
            I[k, q]  = -1 / kind * np.log(np.sqrt(np.dot(np.dot(Fsp[:, q].H, EEG_X_cov[:,k*num_channel:(k+1)*num_channel]), Fsp[:, q])))\
                      - 3/16*(1/kind*(np.dot(np.dot(Fsp[:, q].H, EEG_X_cov[:,k*num_channel:(k+1)*num_channel]), Fsp[:, q]) ** 2 - 1)) ** 2
        I_sort[k, :] = sorted(I[k, :])  # 从小到大排列
        I_Id[k, :]   = np.argsort(I[k, :])


    # 统计m的前几项和，找出极值
    # 取前d个最大的I还原源
    d     = int(num_channel/kind)
    X_Fsp = np.zeros((d, num_channel*kind))
    EEG_S = np.zeros((d, num_samples*kind))
    for k in range(kind):
        for q in range(d):
            X_Fsp[q, k*num_channel:(k+1)*num_channel] = Fsp[int(I_Id[k, num_channel-q-1]), :]
        EEG_S[:, k*num_samples:(k+1)*num_samples] = X_Fsp[:, k*num_channel:(k+1)*num_channel] * EEG_X[:, k*num_samples:(k+1)*num_samples]

    return X_Fsp, EEG_S, d


def test_set(X_Fsp, EEG_X, d, kind):
    """
    此函数对数据进行处理，得到测试集
    输入： X_Fsp：过滤器
          EEG_X：观测到的信号
          d：过滤后的通道数
          kind：类别数
    输出：Vector_sort:测试集
          Labels：对应的标签
    """

    num_channel, num_samples = EEG_X.shape      # num_channel代表通道，num_samples代表总样本数量
    num_samples              = int(num_samples/kind)
    
    # 计算源分量，并计算对应的特征向量
    EEG_S = np.zeros((d, num_samples*kind*kind))
    for k in range(kind):
        for l in range(kind):
            EEG_S[:, (kind*k+l)*num_samples:(kind*k+l+1)*num_samples] = X_Fsp[:, l*num_channel:(l+1)*num_channel]*EEG_X[:, k*num_samples:(k+1)*num_samples]
    
    # 计算方差
    EEG_S_Vars = np.zeros((d, kind*kind))
    for k in range(kind):
        for l in range(kind):
            for p in range(d):
                EEG_S_Vars[p, kind*k+l] = np.var(EEG_S[p, (kind*k+l)*num_channel:(kind*k+l+1)*num_channel])

    # 特征向量计算
    EEG_S_Vars_sum = np.zeros((d, kind))
    Vector         = np.zeros((d, kind*kind))
    for k in range(kind):
        EEG_S_Vars_sum[:, k] = np.sum(EEG_S_Vars[:, k*num_channel:(k+1)*num_channel])
    for k in range(kind):
        for l in range(kind):
            for p in range(d):
                Vector[p, kind*k+l] = np.log(EEG_S_Vars[p, kind*k+l]/EEG_S_Vars_sum[p, k])

    # 对应的标签矩阵
    """
    标签矩阵是一个d × kind*kind的矩阵，每kind列一个单位，为一个类，对应行为标注为1，其余为-1
    """
    Labels = np.zeros((d, kind*kind))
    for k in range(kind):
        lab       = np.ones((d, 3))*(-1)
        lab[:, k] = 1
        Labels[:, kind*k:kind*(k+1)] = lab

    return Vector, Labels
