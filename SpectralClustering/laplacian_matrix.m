function [L,K] = laplacian_matrix(A,g)
% K=exp(-(pdist2(A,B,'squaredeuclidean')/(2*g^2)));
K= rbf_kernel(A,A,g);
K1=K-eye(size(K));
W=zeros(size(K,1));
for i=1:size(W,1)
    [~,index]=sort(K1(i,:),'descend');
     id=index(1:15);
    W(i,id)=K1(i,id);
end

eW = ones(size(W,1),1);
D1 = diag((W*eW).^(-0.5));
% D2 = diag(eW'*W).^-0.5;
L = D1*W*D1;
end
