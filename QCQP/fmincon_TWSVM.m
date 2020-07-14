function [acc_test] = fmincon_TWSVM(dataset,train_test_ratio)
c1 = 100;
c2 = 100;
ep1 = .2;
ep2 = 1;
%%%%[w] = qcqpTWSVM(A,B,c1,c2,ep)
[m,n] = size(dataset);
dataset1 = dataset(randperm(m),:);
num_train = floor(m*train_test_ratio);
train = dataset1(1:num_train,:);
test = dataset1(num_train+1:end,:);
test_feature = [test(:,1:n-1) ones(size(test,1),1)];
[classes] = seprate_class(train);
num_class = length(classes);

y_train = 2*(train(:,n)==-1)-1;%positive class (1) and negative class(-1)
y_test = 2*(test(:,n)==-1)-1;

A = classes(1).data;
B=classes(2).data;
for i=3:num_class
    B = [B;classes(i).data];
end

[m1,d]=size(A);
[m2,~]=size(B);
lb = [-inf*ones(d+1,1);zeros(m1+m2,1)];
x0 = rand(d+1+m1+m2,1);

% x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)
% [f] = objfun(x,d,m1,m2,c1,c2)
% function [cineq,ceq] = nonlcon(x,A,B,ep1,ep2)
x = fmincon(@(x)objfun(x,d,m1,m2,c1,c2),x0,[],[],[],[],lb,[],@(x)nonlcon(x,A,B,ep1,ep2));
betab = x(1:d+1);

ytest_p = sign(-(test_feature*betab).^2+((ep1+ep2)/2));
acc_test = sum(y_test==ytest_p)/size(y_test,1);

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
Xvec = [reshape(X1,[],1) reshape(X2,[],1) ones((step+1)^2,1)];
Y = sign(-(Xvec*betab).^2+((ep1+ep2)/2));
Y = reshape(Y,[step+1,step+1]);
v = [1, 10^-5];
contour(X1,X2,Y,'ShowText','on');
end