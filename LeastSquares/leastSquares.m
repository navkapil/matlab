function [err] = leastSquares(dataset, fit_poly_deg)
%leastSquares(dataset, 3)
if (size(dataset,2)>3)
    disp('more than one feature, maybe going to kernel version may be better');
    %first dimension is input features
    %2nd is noisy y available for training
    %3rd is actual function
end
xtrain = dataset(:,1:end-2);
yact = dataset(:,end);
ytrain = dataset(:,end-1);
[~,I] = sort(xtrain);
figure;
hold on;
plot(xtrain(I),yact(I),'b-');

x = xtrain;
X=[];
x1 = ones(size(x));
for i=0:fit_poly_deg
    X = [x1 X];
    x1 = x1.*x;
end
    
c = (X'*X)\X'*ytrain;
ypred = X*c;
plot(xtrain(I),ypred(I),'r-');
plot(xtrain,ytrain,'go');
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