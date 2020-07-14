function [alpha, b] = fit_kernel_SVM(trainX,trainY,knl,knl_para,C)
EPS = 10^-16;
K = knl(trainX,trainX,knl_para);
Q = trainY'*K*trainY;
e = ones(size(trainX,1),1);
f = -e;
Aeq = trainY';
beq = 0;
lb = 0*e;
ub = C*e;
[alpha] = quadprog(H,f,[],[],Aeq,beq,lb,ub);
sv = alpha>EPS;
alpha1 = zeros(size(alpha));
alpha1 = aplha(sv);
b = e'*(K*trainY*alpha1-trainY*e)/sum(sv);
end