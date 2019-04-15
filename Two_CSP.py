import numpy as np

def two_csp(EEG_X1, EEG_X2, kind = 2):
    """
    针对两类 CSP问题
    EEG_X1, EEG_X2分别是一个 n×T 的numpy数据   EEG_X = A*S
    A是一个满秩的未知 n × n 矩阵。
    S一个 n×T 数据矩阵(源信号)具有以下性质：
        a) 对于每个t, S(:，t)的分量在统计上是独立的；
        b) 对于每个p, S(p，:)是零均值“源信号”的实现。
    特征矩阵的联合近似对角化

    输入 :
            EEG_X1, EEG_X2: 它们的每一列都是n个传感器的样本
            kind: 默认两类任务
    输出 :
            EEG_X1_Fsp, EEG_X2_Fsp: 各自的过滤器
            EEG_S1, EEG_S2：分别是估计源信号
    """

    EEG_X1 = np.mat(np.array(EEG_X1, dtype=np.complex128))
    EEG_X2 = np.mat(np.array(EEG_X2, dtype=np.complex128))
    num_channel, num_samples = EEG_X1.shape      # num_channel代表通道，num_samples代表总样本数量
    channel = range(num_channel)
    samples = range(num_samples)

    ## 零均值
    EEG_X1_zeros = np.zeros((num_channel, num_samples))
    EEG_X2_zeros = np.zeros((num_channel, num_samples))
    for p in channel:
        for q in samples:
            EEG_X1_zeros[p, q] = EEG_X1[p, q] - np.mean(EEG_X1[p])
            EEG_X2_zeros[p, q] = EEG_X2[p, q] - np.mean(EEG_X2[p])

    ## 白化处理

    # 计算各自的的协方差矩阵; 并计算混合协方差矩阵
    EEG_X1_cov = np.dot(EEG_X1_zeros, np.mat(EEG_X1_zeros).H) / np.trace(np.dot(EEG_X1_zeros, np.mat(EEG_X1_zeros).H))
    EEG_X2_cov = np.dot(EEG_X2_zeros, np.mat(EEG_X1_zeros).H) / np.trace(np.dot(EEG_X2_zeros, np.mat(EEG_X1_zeros).H))

    EEG_X_mixcov = EEG_X1_cov + EEG_X2_cov
    EEG_X_mixcov = EEG_X_mixcov / kind

    # 混合协方差矩阵特征分解
    vals, Vecs = np.linalg.eig(EEG_X_mixcov)
    Vals       = np.mat(np.diag(vals))

    # 得到白化矩阵
    white = np.dot(np.sqrt(Vals).I, Vecs.H)

    ## 对X的协方差矩阵进行变换
    #EEG_X1_white = np.dot(np.dot(white, EEG_X1_cov), white.H)
    EEG_X2_white = np.dot(np.dot(white, EEG_X2_cov), white.H)

    # 对转换后的矩阵EEG_X2_white进行对角化处理
    EEG_X2_white_vals, EEG_X_white_Vecs = np.linalg.eig(EEG_X2_white)

    # 对特征值进行排序
    sort_vals = sorted(EEG_X2_white_vals)  # 从小到大
    Id        = np.argsort(EEG_X2_white_vals)  # 排序之前的序号

    # 特征向量也进行相同的变换
    Giv = EEG_X_white_Vecs[:, Id[0]:Id[num_channel - 1] + 1]

    # 此处利用中位数来分割两个特征
    # Ip = np.where(sort_vals > np.median(sort_vals))[0][0]

    # 特征向量分割
    d = int(num_channel/2-0.1)
    U1 = Giv[:, 0:d+1]
    U2 = Giv[:, d:num_channel]

    ## 计算空间滤波器
    EEG_X1_Fsp = np.dot(U1.H, white)
    EEG_X2_Fsp = np.dot(U2.H, white)

    ## 还原源
    EEG_S1 = np.dot(EEG_X1_Fsp, EEG_X1)
    EEG_S2 = np.dot(EEG_X2_Fsp, EEG_X2)

    return EEG_X1_Fsp, EEG_X2_Fsp, EEG_S1, EEG_S2


def test_set(EEG_X1, EEG_X2, EEG_X1_Fsp, EEG_X2_Fsp):
    """
    此函数对数据进行处理，得到测试集
    输入： EEG_X1_Fsp, EEG_X2_Fsp：过滤器
          EEG_X1, EEG_X2：观测到的信号
    输出：Vector_sort:测试集
          Labels：对应的标签
    """

    # 计算源分量，并计算对应的特征向量
    EEG_S11 = np.dot(EEG_X1_Fsp, EEG_X1)
    EEG_S12 = np.dot(EEG_X2_Fsp, EEG_X1)
    EEG_S21 = np.dot(EEG_X1_Fsp, EEG_X2)
    EEG_S22 = np.dot(EEG_X2_Fsp, EEG_X2)
   
    # 计算方差
    num_channel, num_samples = EEG_S11.shape      # num_channel代表通道，num_samples代表总样本数量
    EEG_S11_Vars = np.zeros((num_channel, 1))
    EEG_S12_Vars = np.zeros((num_channel, 1))
    EEG_S21_Vars = np.zeros((num_channel, 1))
    EEG_S22_Vars = np.zeros((num_channel, 1))
    for k in range(num_channel):
        EEG_S11_Vars[k,1] = np.var(EEG_S11[k,:])
        EEG_S12_Vars[k,1] = np.var(EEG_S12[k,:])
        EEG_S21_Vars[k,1] = np.var(EEG_S21[k,:])
        EEG_S22_Vars[k,1] = np.var(EEG_S22[k,:])

    # 归一化处理，特征向量计算
    EEG_S1_Vars = EEG_S11_Vars + EEG_S12_Vars
    EEG_S2_Vars = EEG_S21_Vars + EEG_S22_Vars
    Vector      = np.zeros((num_channel*2, 2))
    
    Vector[0:num_channel, 1]             = np.log(EEG_S11_Vars/EEG_S1_Vars)
    Vector[0:num_channel, 2]             = np.log(EEG_S12_Vars/EEG_S1_Vars)
    Vector[num_channel:num_channel*2, 1] = np.log(EEG_S11_Vars/EEG_S1_Vars)
    Vector[num_channel:num_channel*2, 2] = np.log(EEG_S11_Vars/EEG_S1_Vars)

    # 对应的标签矩阵
    '''1代表第一类，-1代表第二类'''
    Labels = np.zeros((num_channel*2, 1))
    Labels[0:num_channel, 1]             = 1
    Labels[num_channel:num_channel*2, 1] = -1

    return Vector, Labels

# import mne
# from mne.io import concatenate_raws, read_raw_edf
# EEG_X1 = read_raw_edf('C:\\Users\\Administrator\\Desktop\\S001R01.edf', preload=True, stim_channel='auto').to_data_frame().values.transpose()
# EEG_X2 = read_raw_edf('C:\\Users\\Administrator\\Desktop\\S001R02.edf', preload=True, stim_channel='auto').to_data_frame().values.transpose()
