% Non-Linear version of LapLSKSVCR

%% cleaning the workspace
close all;
clear all;

%% Loading of the dataset
num = 500;
cen1 = [0,0];cen2 = [1,0]; cen3 = [2,0];
% points corresponding to cen1 are -ve class
% points corresponding to cen2 are 0 or universum class
% points corresponding to cen3 are +ve class
[dataset]= threecluster(num, cen1, cen2, cen3);

%% creating 3 portions (labelled unlabelled and test sets) of dataset
samples = size(dataset,1);
dataset = dataset(randperm(samples),:);% randomize the data
TrTr = 0.9; % Train data fraction of total rest is Test
train_num = round(0.9*samples);
test_num = samples-train_num;
train_set = dataset(1:train_num,:);
test_set = dataset(train_num+1:end,:);

LblUbl = 0.7; % %age of training data is unlabeled
labld_num = round(LblUbl*train_num);
unlabld_num = train_num-labld_num;
labld_set = train_set(1:labld_num,:);
unlabld_set = train_set(labld_num+1:end,:);

%% prepartion to call training method

% load("threecluster_lineeq.mat");
C = seprate_class(labld_set);
c1 = 100;
c2 = 10;
c3 = 5;
delta = 0.8; %value must be between 0 & 1
mu = 0.2;
knl = @rbf_kernel;
[u,M] = NonLinLapLSKSVCR(C(1).data,C(2).data,C(3).data,unlabld_set(:,1:end-1),c1,c2,c3,mu,delta,knl,mu);
% C(1)=-1(m),C(2)=0(l),C(3)=1 (f)
% [u,S1] = NonLinLapLSKSVCR(B,C,A,U, c1,c3,c5,mu,delta,knl,knl_para)

%% prediction
% u1 = u1/norm(u1);
% u2 = u2/norm(u2);
% u3 = u1+u2/norm(u1+u2);


k_testf = knl(test_set(:,1:end-1),M,mu);
test_f = [k_testf ones(size(k_testf,1),1)];
d = test_f*u;

I3 = d > delta;
I1 = d < -delta;
% IZ = ones(size(test,1),1)-or(IP,IN);
PC = I3-I1;
acc = sum(PC==test_set(:,end))/size(test_set,1);

%% plotting contours 
NonLin_plot_decision_bndry(u,train_set(:,1:end-1),train_set(:,end),knl,M,mu)
