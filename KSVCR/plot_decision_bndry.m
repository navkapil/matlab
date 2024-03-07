function [orient] =  plot_decision_bndry(model)
%function will plot with respect to first two feature attributes
step =100;
figure(1)
hold on

minx1 = min(model.P(:,1));maxx1=max(model.P(:,1));
minx2 = min(model.P(:,2));maxx2=max(model.P(:,2));
x1 = (minx1:(maxx1-minx1)/(step-1):maxx1)';
x2 = (minx2:(maxx2-minx2)/(step-1):maxx2)';
[X1,X2] = meshgrid(x1,x2);
Xvec = [reshape(X1,[],1) reshape(X2,[],1)];

[~,~,score,orient] = test_KSVCR(model,Xvec,[]);

Y = reshape(score,[step,step]);
v = [-1,-(model.delta+1)/2,0,(model.delta+1)/2,1];
contour(X1,X2,Y,v,'ShowText','on');
end