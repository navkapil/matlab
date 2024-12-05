function [ker] = rbf_kernel(A,B,mu)
    if(nargin<3)
        mu = 1;
    end
    m1 = size(A,1);
    m2 = size(B,1);
    ker = zeros(m1,m2);
    for i=1:m1
        for j=1:m2
            nom = norm(A(i,:)-B(j,:));
            ker(i,j) = exp(-mu*nom*nom);
        end
    end            
end