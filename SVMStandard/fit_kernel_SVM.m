function [alpha, b] = fit_kernel_SVM(trainX,trainY,knl,knl_para,C)
EPS = 10^-16;
K = knl(trainX,trainX,knl_para);
Q = diag(trainY)*K*diag(trainY);
e = ones(size(trainX,1),1);
f = -e;
Aeq = trainY';
beq = 0;
lb = 0*e;
ub = C*e;
alpha = quadprog(Q,f,[],[],Aeq,beq,lb,ub);
sv = alpha>EPS;
alpha(~sv)=0;
b = e'*(K*diag(trainY)*alpha-diag(trainY)*e)/sum(sv);
end