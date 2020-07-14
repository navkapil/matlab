function [f] = objfun(x,d,m1,m2,c1,c2)
beta = x(1:d);
xsi1 = x(d+2:d+1+m1);
xsi2 = x(d+1+m1+1:d+1+m1+m2);

f = 0.5*(beta'*beta)+ c1*ones(1,m1)*xsi1+c2*ones(1,m2)*xsi2;
end
