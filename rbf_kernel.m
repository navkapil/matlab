function [ker] = rbf_kernel(A,B,mu)
% function or kernel returns similarity between two vectors(between 0 to 1), the first one is a row of A another
% is a row of B, higher value of mu quickly reduces the similarity value to zero as the two vectors go farther 
% and lower value of mu, flattens the similarity curve, giving rise to more smoother boundaries.
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
