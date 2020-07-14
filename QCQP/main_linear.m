function [y_testp,y_trainp,acc_test,acc_train] = main_linear(dataset, train_test_ratio,alpha)
c1 = 100;
c2 = 100;
ep = .01;
%%%%[w] = qcqpTWSVM(A,B,c1,c2,ep)
[m,n] = size(dataset);
dataset1 = dataset(randperm(m),:);
num_train = floor(m*train_test_ratio);
train = dataset1(1:num_train,:);
test = dataset1(num_train+1:end,:);

[classes] = seprate_class(train);
num_class = length(classes);

y_train = (train(:,n)==-1);%positive class (1) and negative class(0)
y_test = (test(:,n)==-1);

A = classes(1).data;
B=classes(2).data;
for i=3:num_class
    B = [B;classes(i).data];
end

[w] = qcqpTWSVM([A ones(size(A,1),1)],[B ones(size(B,1),1)],c1,c2,ep,alpha);


%% predictions

test_data = [test(:,1:n-1) ones(size(test,1),1)];
% lbl_pos = classes(1).label;

% prediction for test data
y_testp = zeros(size(test_data,1),1);
y_testp(abs(test_data*w)<ep)=1;
acc_test = sum(y_test==y_testp)/size(y_test,1);

% prediction for train data
train_feature = [train(:,1:n-1) ones(size(train,1),1)];
y_trainp = zeros(size(train_feature,1),1);
y_trainp(abs(train_feature*w)<ep)=1;
acc_train = sum(y_train==y_trainp)/size(y_train,1);

%% plotting business
figure(1);
hold on;

%plotting orginal data
pos_test = (y_test==1);
plot(test(pos_test,1),test(pos_test,2),'b.',test(~pos_test,1),test(~pos_test,2),'r.');

%plotting predictions
pos_testp = (y_testp==1);
plot(test(pos_testp,1),test(pos_testp,2),'bo',test(~pos_testp,1),test(~pos_testp,2),'ro');

%contour plot
minx1 = min(dataset(:,1));maxx1=max(dataset(:,1));
minx2 = min(dataset(:,2));maxx2=max(dataset(:,2));
x1 = (minx1:(maxx1-minx1)/10:maxx1)';
x2 = (minx2:(maxx2-minx2)/10:maxx2)';
[X1,X2] = meshgrid(x1,x2);
Xvec = [reshape(X1,[],1) reshape(X2,[],1) ones(121,1)];
Y = Xvec*w;
Y = reshape(Y,[11,11]);
v = [1, 10^-5];
contour(X1,X2,Y,'ShowText','on');
end