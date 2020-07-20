function [err] = knl_leastSquares(dataset, knl_para)
% The code is kernelized with the help of http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/lecturenote14_2.html
%knl_leastSquares(dataset, 2)

xtrain = dataset(:,1:end-2);
yact = dataset(:,end);
ytrain = dataset(:,end-1);
[~,I] = sort(xtrain(:,1));
figure;
hold on;
plot(xtrain(I,1),yact(I),'b-');

x = xtrain;
K = rbf_kernel(x,x,knl_para);
c = (K+0.01*eye(size(K)))\ytrain;% 0.01*eye(size(K)) Tikhnovs regularization term
xtest = xtrain(I);%(-5:0.1:5)';
ypred = rbf_kernel(xtest,x,knl_para)*c;


plot(xtest,ypred,'r-');
plot(xtrain(:,1),ytrain,'go');
hold off
err = norm(ypred-yact)/max(yact);
end
% K = rbf_kernel(x,x);
% K = rbf_kernel(x,x,0.1);
% alpha = K\dataset(:,2);
% xtest = (-5:0.1:5)';
% ypred = rbf_kernel(xtest,x,0.1)*alpha;
% yact = [xtest.^3 xtest.^2 xtest xtest.^0]*oricoeff;
% plot(xtest,yact,'r-',xtest,ypred,'b-',
% plot(xtest,yact,'r-',xtest,ypred,'b-')
