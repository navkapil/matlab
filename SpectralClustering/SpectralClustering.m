function []=SpectralClustering(data,knl_para,k)
%SpectralClustering(data,knl_para,k)
% data with samples in each row and features/attributes in each column
%k -clusters
%knl_para kernel parameter

[L,K] = laplacian_matrix(data,knl_para);
[Vec,Val]=eig((L+L')/2);
m = size(Val,2);
X = Vec(:,m-k+1:m);
Y = zeros(size(X));
for i=1:size(X,1)
    Y(i,:) = X(i,:)/norm(X(i,:));
end
[idx,C] = kmeans(Y,k);
figure;
PlotStyle={'r.','b.','m.','g.'};
hold on
for i=1:k
    testPoint=idx==i;
    xx = data(testPoint,1);
    yy=data(testPoint,2);
    ps = PlotStyle{i};
    plot(xx,yy,ps);
end
end