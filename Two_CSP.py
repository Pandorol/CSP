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
    Ip = np.where(sort_vals > np.median(sort_vals))[0][0]

    # 特征向量分割
    U1 = Giv[:, 0:Ip]
    U2 = Giv[:, Ip:num_channel]

    ## 计算空间滤波器
    EEG_X1_Fsp = np.dot(U1.H, white)
    EEG_X2_Fsp = np.dot(U2.H, white)

    ## 还原源
    EEG_S1 = np.dot(EEG_X1_Fsp, EEG_X1)
    EEG_S2 = np.dot(EEG_X2_Fsp, EEG_X2)

    return EEG_X1_Fsp, EEG_X2_Fsp, EEG_S1, EEG_S2


# import mne
# from mne.io import concatenate_raws, read_raw_edf
# EEG_X1 = read_raw_edf('C:\\Users\\Administrator\\Desktop\\S001R01.edf', preload=True, stim_channel='auto').to_data_frame().values.transpose()
# EEG_X2 = read_raw_edf('C:\\Users\\Administrator\\Desktop\\S001R02.edf', preload=True, stim_channel='auto').to_data_frame().values.transpose()
# two_csp(EEG_X1, EEG_X2)
# EEG_X1 = np.random.rand(5,100)
# EEG_X2 = np.random.rand(5,100)
# two_csp(EEG_X1, EEG_X2)
