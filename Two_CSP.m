function [EEG_S1, EEG_S2] = Two_CSP(EEG_X1, EEG_X2)

%% ************************************************************************
% �������CSP����
% EEG_X1, EEG_X2�ֱ���һ�� n��T ������ EEG_X = A S
%   A��һ�����ȵ�δ֪ n �� n ����
% 	Sһ�� n��T ���ݾ���(Դ�ź�)�����������ʣ�
%    	a)����ÿ��t, S(:��t)�ķ�����ͳ�����Ƕ�����
%       b)����ÿ��p, S(p��:)�����ֵ��Դ�źš���ʵ�֡�
%
% ������������Ͻ��ƶԽǻ�
%
% ���� :
%   * EEG_X1, EEG_X2: ���ǵ�ÿһ�ж���n��������������
% ��� :
%     EEG_S1, EEG_S2�ֱ�����һ�� n��T ��ԭʼ(��pinv(A)��x)����Դ�ź�
%**************************************************************************
kind   = 2;  % ��������
[n, T] = size(EEG_X1);
t = 1:T;
%% ���ֵ
for p = 1:n
    for q = 1:T
        X1(p, q) = EEG_X1(p, q) - mean(EEG_X1(p, :));
        X2(p, q) = EEG_X2(p, q) - mean(EEG_X2(p, :));
    end
end
%% �׻�����
% ������Եĵ�Э������� ��������Э�������
R = zeros(n, n);
R1 = X1*X1'/trace(X1*X1');
R2 = X2*X2'/trace(X2*X2');
R = R1 + R2;
R = R/kind;
% ���Э������������ֽ�
[U, D] = eig(R);
% �õ��׻�����
P = D^(-1/2) * U';
% ��X��Э���������б任
Y1 = P * R1 *P';
Y2 = P * R2 *P';

%% ��ת����ľ���Y���жԽǻ�����
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
%% ����ռ��˲���
F1 = U1'*P;
F2 = U2'*P;
% ��ԭԴ
EEG_S1 = F1*X1;
EEG_S2 = F2*X2;
%% ��ͼ
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
