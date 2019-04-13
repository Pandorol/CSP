function EEG_S = Multi_CSP(EEG_X)
%% ************************************************************************
% Jacobi算法针对多类CSP问题
% EEG_X是一个元胞，里面的分量x
%   x是一个 n×T 的数据 EEG_X = A S
%   A是一个满秩的未知 n × n 矩阵。
% 	S一个 n×T 数据矩阵(源信号)具有以下性质：
%    	a)对于每个t, S(:，t)的分量在统计上是独立的
%       b)对于每个p, S(p，:)是零均值“源信号”的实现。
%
% 特征矩阵的联合近似对角化
%
% 输入 :
%   * EEG_X: EEG_X分量的每一列都是n个传感器的样本
% 输出 :
%    * EEG_S是一个 n×T 的原始(即pinv(A)×x)估计源信号
%**************************************************************************
kind = length(EEG_X);
[n, T] = size(EEG_X{1,1});
t = 1:T;
m = n;
%% 零均值
for k = 1: kind
    for p = 1:n
        for q = 1:T
            X{1, k}(p, q) = EEG_X{1, k}(p, q) - mean(EEG_X{1, k}(p, :));
        end
    end
end
%% 白化处理
% 计算X分量的协方差 并计算混合协方差矩阵
R = zeros(n, n);
R1 = cell(1, kind);
for k = 1:kind
    R1{1, k} = X{1, k}*X{1, k}'/trace(X{1, k}*X{1, k}');
    R = R + R1{1, k};
end
R = R/kind;
% 混合协方差矩阵特征分解
[U, D] = eig(R);
% 得到白化矩阵
P = D^(-1/2) * U';
% 对X的协方差矩阵进行变换
for k = 1:kind
    Y{1, k} = P * R1{1, k} *P';
end
M = Y; % M对角化后的对角阵

%% 对转换后的矩阵Y进行对角化处理
% Jacobi多矩阵近似联合对角化
Giv = eye(n);
amin = 10;
mm = 0;
while amin > 25
    amin = 0;
    mm = mm + 1
    p = randint(1,1,[1 n-1])
    q = randint(1,1,[p+1 n])
    h = zeros(1, 3);
    for k = 1:kind
        h = h + [M{1,k}(p, p)-M{1,k}(q, q), M{1,k}(p, q)+M{1,k}(q, p), 1i*(M{1,k}(q, p)-M{1,k}(p, q))];
    end
    [vcp, D] = eig(real(h'*h));
    [la, K]	= sort(diag(D));
    angles = vcp(:, K(3));
    % 防止出现零
    if angles(1) < 0
        angles = -angles;
    end
    r = sqrt(abs(angles(1))^2+abs(angles(2))^2+abs(angles(3))^2);
    c = sqrt((angles(1)+r)/(2*r));
    s = ((angles(2)-1i*angles(3))/sqrt(2*r*(angles(1)+r)));
    %%% 通过Givens旋转更新矩阵M和V
    G = eye(m);
    G(p, p) = c;
    G(p, q) = conj(s);
    G(q, p) = -s;
    G(q, q) = conj(c);
    Giv = Giv*G;
    for k = 1:kind
        M{1, k} = G' * M{1,k} * G;
        amin = amin + sum(sum(abs(M{1,k} - diag(diag(M{1,k}))).^2));
    end % for 
    amin
end % if

%% 得到特征向量Giv
%% 计算空间滤波器
F = Giv'*P;
% 计算I,寻找最大的值
for p = 1:kind
    for q = 1:n
        I{1, p}(q) = -1/kind*log(sqrt(F(:,q)'*R1{1,p}*F(:,q))) - 3/16*(1/kind*((F(:,q)'*R1{1,p}*F(:,q))^2-1))^2;
    end
    [x, y] = sort(I{1, p});
end
%%统计m的前几项和，找出极值
% 取前d个最大的I  还原源
d = n;
for p = 1:kind
    for q = 1:d
        Fcsp{1, p}(q,:) = F(y(n+1-q),:);
    end
    EEG_S{1, p} = Fcsp{1, p}*EEG_X{1, p};
end
%% 画图
for p = 1:kind
    for q = 1:d
        figure(p)
        subplot(d,1,q)
        plot(t, EEG_S{1,p}(q,:))
    end
end
