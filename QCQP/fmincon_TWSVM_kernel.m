function [acc_test,acc_train] = fmincon_TWSVM_kernel(dataset,train_test_ratio)
c1 = 100;
c2 = 100;
ep1 = 1;
ep2 = 10;
g=2;
%%%%[w] = qcqpTWSVM(A,B,c1,c2,ep)
[m,n] = size(dataset);
dataset1 = dataset(randperm(m),:);
num_train = floor(m*train_test_ratio);
train = dataset1(1:num_train,:);
test = dataset1(num_train+1:end,:);

% [train,mn,mx] =  normaltrain(train,[0,1]);
% [test] = normaltest(test,mn,mx,[0,1]);

test_feature = test(:,1:n-1);

[classes] = seprate_class(train);
num_class = length(classes);

y_train = 2*(train(:,n)==-1)-1;%positive class (1) and negative class(-1)
y_test = 2*(test(:,n)==-1)-1;

A = classes(1).data;
B=classes(2).data;
for i=3:num_class
    B = [B;classes(i).data];
end


KA = rbf_kernel(A,[A;B],g);
KB = rbf_kernel(B,[A;B],g);
Ktest = [rbf_kernel(test_feature,[A;B],g) ones(size(test,1),1)];
Ktrain = [rbf_kernel(train(:,1:end-1),[A;B],g) ones(size(train,1),1)];

[m1,d]=size(KA);
[m2,~]=size(KB);
lb = [-inf*ones(d+1,1);zeros(m1+m2,1)];
x0 = rand(d+1+m1+m2,1);

% x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)
% [f] = objfun(x,d,m1,m2,c1,c2)
% function [cineq,ceq] = nonlcon(x,A,B,ep1,ep2)
options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',2000);
x = fmincon(@(x)objfun(x,d,m1,m2,c1,c2),x0,[],[],[],[],lb,[],@(x)nonlcon(x,KA,KB,ep1,ep2),options);
betab = x(1:d+1);

temp1 = -(Ktest*betab).^2;
temp2 = temp1+((ep1+ep2)/2);
ytest_p = sign(temp2);
acc_test = sum(y_test==ytest_p)/size(y_test,1);

temp1 = -(Ktrain*betab).^2;
temp2 = temp1+((ep1+ep2)/2);
ytrain_p = sign(temp2);
acc_train = sum(y_train==ytrain_p)/size(y_test,1);
%% plotting business
figure(1);
hold on;

%plotting orginal data
pos_test = (y_test==1);
plot(test(pos_test,1),test(pos_test,2),'b.',test(~pos_test,1),test(~pos_test,2),'r.');

%plotting predictions
pos_testp = (ytest_p==1);
plot(test(pos_testp,1),test(pos_testp,2),'bo',test(~pos_testp,1),test(~pos_testp,2),'ro');

%contour plot
step = 10;
minx1 = min(dataset(:,1));maxx1=max(dataset(:,1));
minx2 = min(dataset(:,2));maxx2=max(dataset(:,2));
x1 = (minx1:(maxx1-minx1)/step:maxx1)';
x2 = (minx2:(maxx2-minx2)/step:maxx2)';
[X1,X2] = meshgrid(x1,x2);
Xvec = [reshape(X1,[],1) reshape(X2,[],1)];
KXvec = [rbf_kernel(Xvec,[A;B],g) ones((step+1)^2,1)];
temp1 = -(KXvec*betab).^2;
Y = temp1+((ep1+ep2)/2);
% Y = sign(temp2;
Y = reshape(Y,[step+1,step+1]);
v = [1, 10^-5];
contour(X1,X2,Y,'ShowText','on');
end