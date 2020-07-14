function plot_decision_bndry(alpha,b,trainX,trainY,knl,knl_para)
%function will plot with respect to first two feature attributes
[m,n]=size(trainX);


step =20;
figure(1)
minx1 = min(trainX(:,1));maxx1=max(trainX(:,1));
minx2 = min(trainX(:,2));maxx2=max(trainX(:,2));
x1 = (minx1:(maxx1-minx1)/(step-1):maxx1)';
x2 = (minx2:(maxx2-minx2)/(step-1):maxx2)';
[X1,X2] = meshgrid(x1,x2);
Xvec = [reshape(X1,[],1) reshape(X2,[],1)];
Y = rbf_kernel(Xvec,M,g)*beta;
Y = reshape(Y,[(step+1),(step+1)]);
v = [1, 10^-5];
contour(X1,X2,Y,'ShowText','on');

mn = min();
mx = max(trainX);
grid_points = [100,100];
X1 = mn(1):(mx(1)-mn(1))/(grid_points(1)-1):mx(1);
X2 = mn(2):(mx(2)-mn(2))/(grid_points(2)-1):mx(2);
[XX1 XX2] = meshgrid(X1,X2);

K_plot = knl([X1 X2]

end