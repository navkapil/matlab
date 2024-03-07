function [model] = KSVCR(B,C,A,c1,c2,delta,mu,kernel)
% Code for Moosaei,H.,and Haldik, M. (2021) "Least Squares approach to 
% K-SVCR multi-class classification with its application"  Section 2.4 
% K-Support vector classification regression (Wolfe's Dual Implemetation)
% Assuming A matrix has features for positive class 
%          B matrix has features for negative class 
%          C matrix has features for none class or Universum
%          c1,c2 are different penalty parameters
%          delta zero class separation parameter 
%          testdata is labelled data for getting accuracy and prediction
        
    [np,~] = size(A);
    [nn,~] = size(B);
    [nz,~] = size(C);
    
    lb = zeros(np+nn+2*nz,1);
    e1 = ones(np,1);
    e2 = ones(nn,1);
    e3 = ones(nz,1);
    q = [e1;e2;-delta*e3;-delta*e3];
    F = [c1*e1;c1*e2;c2*e3;c2*e3];
    P=[A;B;C;C];
    J = diag([e1;-e2;e3;-e3]);
    
    % kernel version
    % linear 
    if strcmp(kernel,'linear')
            K = P*P';
    elseif strcmp(kernel,'rbf')
            % rbf
            ker = @rbf_kernel;
            K = ker(P, P, mu);
    end
    
    H = J*K*J';
    gamma = quadprog(H,-q,[],[],[],[],lb,F);
    epsilon = max(gamma,[],'all')*10^-5;
    sv = (gamma>epsilon) & (gamma < F-epsilon);
    b = sum(q(sv)-J(sv,sv)*K(sv,:)*J'*gamma)/sum(sv);

    model.P = P;
    model.J = J;
    model.gamma = gamma;
    model.b = b;
    model.kernel = kernel;
    model.mu = mu;
    model.delta = delta;
end  