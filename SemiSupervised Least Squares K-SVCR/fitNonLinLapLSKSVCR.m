function [Mdl] = fitNonLinLapLSKSVCR(foldableTrainData,otherpara, parameter)
% function [u,M] = fitNonLinLapLSKSVCR(B,C,A,U, c1,c3,c5,mu,delta,knl,knl_para)
% Code for Laplacian K-SVCR (Wolfe's Dual Implemetation) the proposed 
% Non-Linear Laplacian version of H. Moosaei, M. Haldik.Least Squares approach to 
% KSVCR multiclass classification with its applications 's section 2.4
%
% Assuming A matrix has features for positive class 
%          B matrix has features for negative class 
%          C matrix has features for none class or Universum
%          U all unlabelled samples
%          c1 penalty parameter for +ve class
%          c2 penalty parameter for -ve class
%          c3,c4 penalty parameter for neutral class from +ve and -ve class
%          separately
%          c5 is manifold regularization penalty parameters
%          mu graph laplacian kernel parameter
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
    [c1,c3,c5,delta,knl_para] = para{:};



%% for simplifying the tuning of optimal parameters
    c2=c1;
    c4=c3;
%% intilizaing different variables from formulation

    [m1,dim] = size(A);
    [m2,~] = size(B);
    [n,~] = size(C);
    [u,~] = size(U);
    e1 = ones(m1,1);
    e2 = ones(m2,1);
    e3 = ones(n,1);
    eu = ones(u,1);
    M = [A;B;C;U];
    
    %graph laplacian matrix and Kernel matrix calculation
    S3 = knl(M,M,knl_para);

    % if reduced kernel is used
    if isfield(otherpara,'redKerIn')
        redKerIn = otherpara.redKerIn;
    else 
        redKerIn = 1:(m1+m2+n+u);
    end
    

    matSize = length(redKerIn)+1;

    P1 = S3(1:m1,redKerIn);
    Q1 = S3(m1+1:m1+m2,redKerIn);
    R1 = S3(m1+m2+1:m1+m2+n,redKerIn);
    S2 = S3(:,redKerIn);

    P = [P1 e1];
    Q = [Q1 e2];
    R = [R1 e3];
    S = [S2 ones(size(M,1),1)];
    W = S3;
    for i = 1:size(W,1)
        [~,Ind]=sort(W(i,:),'descend');
        W(i,Ind(k+1:end)) = 0;
    end
    e = ones(m1+m2+n+u,1);
    L = diag(0.5*(W*e+W'*e))-W; % laplacian matrix prepared
    
    L1 = (L+L')/2;
    H = speye(matSize)+c1*(P'*P)+c2*(Q'*Q)+(c3+c4)*(R'*R)+c5*(S'*L1*S);
    f = c1*P'*e1-c2*Q'*e2+(c3-c4)*delta*R'*e3;
    Mdl.u = H\f;
    Mdl.M = M;
end