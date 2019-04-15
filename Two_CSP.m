function [EEG_S1, EEG_S2] = Two_CSP(EEG_X1, EEG_X2)

%% ************************************************************************
% 针对两类CSP问题
% EEG_X1, EEG_X2分别是一个 n×T 的数据 EEG_X = A S
%   A是一个满秩的未知 n × n 矩阵。
% 	S一个 n×T 数据矩阵(源信号)具有以下性质：
%    	a)对于每个t, S(:，t)的分量在统计上是独立的
%       b)对于每个p, S(p，:)是零均值“源信号”的实现。
%
% 特征矩阵的联合近似对角化
%
% 输入 :
%   * EEG_X1, EEG_X2: 它们的每一列都是n个传感器的样本
% 输出 :
%     EEG_S1, EEG_S2分别是是一个 n×T 的原始(即pinv(A)×x)估计源信号
%**************************************************************************
kind   = 2;  % 两类任务
[n, T] = size(EEG_X1);
t = 1:T;
%% 零均值
for p = 1:n
    for q = 1:T
        X1(p, q) = EEG_X1(p, q) - mean(EEG_X1(p, :));
        X2(p, q) = EEG_X2(p, q) - mean(EEG_X2(p, :));
    end
end
%% 白化处理
% 计算各自的的协方差矩阵 并计算混合协方差矩阵
R = zeros(n, n);
R1 = X1*X1'/trace(X1*X1');
R2 = X2*X2'/trace(X2*X2');
R = R1 + R2;
R = R/kind;
% 混合协方差矩阵特征分解
[U, D] = eig(R);
% 得到白化矩阵
P = D^(-1/2) * U';
% 对X的协方差矩阵进行变换
Y1 = P * R1 *P';
Y2 = P * R2 *P';

%% 对转换后的矩阵Y进行对角化处理
[UU, DD] = eig(Y2);
[ip, iq] = sort(diag(DD));
for k = 1:n
    AD(k) = DD(iq(k), iq(k));
    Giv(:, k) = UU(:,iq(k));
end
x  = find(AD >= median(AD));
il = x(1);
for k = 1:il
    U1(:, k) = Giv(:, k);
end
for k = 1:n-il
    U2(:, k) = Giv(:, k+il);
end
%% 计算空间滤波器
F1 = U1'*P;
F2 = U2'*P;
% 还原源
EEG_S1 = F1*X1;
EEG_S2 = F2*X2;
%% 画图
figure(1)
for k = 1:il
    subplot(il,1,k)
    plot(t, EEG_S1(k, :))
end
figure(2)
for k = 1:n-il
    subplot(n-il,1,k)
    plot(t, EEG_S2(k, :))
end
