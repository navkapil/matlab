function [Mdl] = fitNonLinLapKSVCR(foldableTrainData,otherpara, parameter)
% [out] = train_func(trainfold,otherpara,parameter(i+1,:))
% Code for Laplacian K-SVCR (Wolfe's Dual Implemetation) the proposed 
% Non-Linear Laplacian version of H. Moosaei, M. Haldik.Least Squares approach to 
% KSVCR multiclass classification with its applications 's section 2.4
%
% Assuming A matrix has features for positive class 
%          B matrix has features for negative class 
%          C matrix has features for none class or Universum
%          U all unlabelled samples
%          k graph laplacian k-nn
%          c1,c2,c3 are different penalty parameters
%          mu graph laplacian kernel parameter--not used--
%          delta zero class separation parameter 
%          knl is kernel function chossen
%          knl_para is kernel parameters
% Author: Kapil

    A = foldableTrainData(3).data(:,1:end-1);
    B = foldableTrainData(1).data(:,1:end-1);
    C = foldableTrainData(2).data(:,1:end-1);

    k = otherpara.graph_k; % Graph laplacian parameter as k-nn
    U = otherpara.U(:,1:end-1);
    knl = otherpara.knl;
    % [c1,c3,delta,knl_para]=deal(0);
    para = num2cell(parameter);
    [c1,c2,c3,delta,knl_para] = para{:};
    % c2 = c1;
    kernelMat = [A;B;C;U];
    %reducedKnlSz = 0.05;
    %redKernelMat = kernelMat(randsample(size(kernelMat,1),round(reducedKnlSz*size(kernelMat,1))));
    A = knl(A,kernelMat,knl_para);
    B = knl(B,kernelMat,knl_para);
    C = knl(C,kernelMat,knl_para);
    
    [m1,dim] = size(A);
    [m2,~] = size(B);
    [n,~] = size(C);
    
    e1 = ones(m1,1);
    e2 = ones(m2,1);
    e3 = ones(n,1);
    ep = [e1;e2];
    en = [e3;e3];
    
    P = [[A;B] ep];
    N = [[C;C] en];
    nu =size(U,1);
%     M1 = [P;[C e3];[U ones(nu,1)]];
    mu = 0.5;
    M = [knl(kernelMat,kernelMat,knl_para) ones(m1+m2+n+nu,1)];
    W = knl(kernelMat,kernelMat,mu);
    D1 = sparse([speye(m1), zeros(m1,m2);zeros(m2,m1), -speye(m2)]);
    D2 = sparse([speye(n), zeros(n);zeros(n), -speye(n)]);
    for i = 1:size(W,1)
        [~,Ind]=sort(W(i,:),'descend');
        W(i,Ind(k+1:end)) = 0;
        W(i,Ind(1:k)) = 1;
    end
    e = ones(m1+m2+n+nu,1);
    L = diag(0.5*(W*e+W'*e))-W; % laplacian matrix prepared
    
    L1 = (L+L')/2;
    Q = c3*((1/c3)*speye(size(M,2))+M'*L1*M);
    S = [P;N];
    D = [D1 zeros(m1+m2,2*n);zeros(2*n,m1+m2) D2];
    c4 = [c1*ep;c2*en];
    H = D*S*(Q\S')*D;
    c5 = [-ep;delta*en];

    lb = zeros(m1+m2+2*n,1);
    ub = c4;

    theta = quadprog(H,c5,[],[],[],[],lb,ub);
    Mdl.u = Q\S'*D*theta;
    Mdl.kernelMat = kernelMat;
end