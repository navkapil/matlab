function plot_decision_bndry(alpha,b,trainX,trainY,knl,knl_para)
%function will plot with respect to first two feature attributes
step =100;
figure(1)
hold on
plus = trainY==1;

plot(trainX(plus,1),trainX(plus,2),'r+');
plot(trainX(~plus,1),trainX(~plus,2),'b.');

minx1 = min(trainX(:,1));maxx1=max(trainX(:,1));
minx2 = min(trainX(:,2));maxx2=max(trainX(:,2));
x1 = (minx1:(maxx1-minx1)/(step-1):maxx1)';
x2 = (minx2:(maxx2-minx2)/(step-1):maxx2)';
[X1,X2] = meshgrid(x1,x2);
Xvec = [reshape(X1,[],1) reshape(X2,[],1)];

%calculation of decision function
K_test = knl(Xvec,trainX,knl_para);
f_test = K_test*diag(trainY)*alpha-b;%decision function

Y = reshape(f_test,[step,step]);
% v = [1, 10^-5];
contour(X1,X2,Y,'ShowText','on');
end