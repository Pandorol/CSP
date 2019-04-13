function EEG_S = Multi_CSP(EEG_X)
%% ************************************************************************
% Jacobi�㷨��Զ���CSP����
% EEG_X��һ��Ԫ��������ķ���x
%   x��һ�� n��T ������ EEG_X = A S
%   A��һ�����ȵ�δ֪ n �� n ����
% 	Sһ�� n��T ���ݾ���(Դ�ź�)�����������ʣ�
%    	a)����ÿ��t, S(:��t)�ķ�����ͳ�����Ƕ�����
%       b)����ÿ��p, S(p��:)�����ֵ��Դ�źš���ʵ�֡�
%
% ������������Ͻ��ƶԽǻ�
%
% ���� :
%   * EEG_X: EEG_X������ÿһ�ж���n��������������
% ��� :
%    * EEG_S��һ�� n��T ��ԭʼ(��pinv(A)��x)����Դ�ź�
%**************************************************************************
kind = length(EEG_X);
[n, T] = size(EEG_X{1,1});
t = 1:T;
m = n;
%% ���ֵ
for k = 1: kind
    for p = 1:n
        for q = 1:T
            X{1, k}(p, q) = EEG_X{1, k}(p, q) - mean(EEG_X{1, k}(p, :));
        end
    end
end
%% �׻�����
% ����X������Э���� ��������Э�������
R = zeros(n, n);
R1 = cell(1, kind);
for k = 1:kind
    R1{1, k} = X{1, k}*X{1, k}'/trace(X{1, k}*X{1, k}');
    R = R + R1{1, k};
end
R = R/kind;
% ���Э������������ֽ�
[U, D] = eig(R);
% �õ��׻�����
P = D^(-1/2) * U';
% ��X��Э���������б任
for k = 1:kind
    Y{1, k} = P * R1{1, k} *P';
end
M = Y; % M�Խǻ���ĶԽ���

%% ��ת����ľ���Y���жԽǻ�����
% Jacobi�����������϶Խǻ�
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
    % ��ֹ������
    if angles(1) < 0
        angles = -angles;
    end
    r = sqrt(abs(angles(1))^2+abs(angles(2))^2+abs(angles(3))^2);
    c = sqrt((angles(1)+r)/(2*r));
    s = ((angles(2)-1i*angles(3))/sqrt(2*r*(angles(1)+r)));
    %%% ͨ��Givens��ת���¾���M��V
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

%% �õ���������Giv
%% ����ռ��˲���
F = Giv'*P;
% ����I,Ѱ������ֵ
for p = 1:kind
    for q = 1:n
        I{1, p}(q) = -1/kind*log(sqrt(F(:,q)'*R1{1,p}*F(:,q))) - 3/16*(1/kind*((F(:,q)'*R1{1,p}*F(:,q))^2-1))^2;
    end
    [x, y] = sort(I{1, p});
end
%%ͳ��m��ǰ����ͣ��ҳ���ֵ
% ȡǰd������I  ��ԭԴ
d = n;
for p = 1:kind
    for q = 1:d
        Fcsp{1, p}(q,:) = F(y(n+1-q),:);
    end
    EEG_S{1, p} = Fcsp{1, p}*EEG_X{1, p};
end
%% ��ͼ
for p = 1:kind
    for q = 1:d
        figure(p)
        subplot(d,1,q)
        plot(t, EEG_S{1,p}(q,:))
    end
end
