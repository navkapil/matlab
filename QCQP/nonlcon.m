function [cineq,ceq] = nonlcon(x,A,B,ep1,ep2)
[m1,d]=size(A);
[m2,~]=size(B);
betab = x(1:d+1);
xsi1 = x(d+2:d+1+m1);
xsi2 = x(d+1+m1+1:d+1+m1+m2);
cons1 = zeros(m1,1);
cons2 = zeros(m2,1);
for i=1:m1
    cons1(i)=(betab'*[A(i,:) 1]'*[A(i,:) 1]*betab)^2-[zeros(i-1,1);1;zeros(m1-i,1)]'*xsi1-ep1;
end
for j=1:m2
    temp1 = -(betab'*[B(j,:) 1]'*[B(j,:) 1]*betab)^2;
    temp2 = [zeros(j-1,1);1;zeros(m2-j,1)]';
    temp2 = temp2*xsi2;
    cons2(j)=temp1-temp2+ep2;
end
cineq = [cons1;cons2];
ceq=[];