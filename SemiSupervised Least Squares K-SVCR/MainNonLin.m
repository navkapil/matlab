%% Driver Program for crossvalidation equipped with reduced kernel version
% of the Semi-Supervised Least Squares K-SVCR Implementation 
% Paper: Vivek Srivastva & Kapil. 2024. Semi-Supervised Laplacian Least Squares K-SVCR for
% Three-Class Classification

%% Author Information:
% Name: Kapil
% Affiliation: NIT Kurukshetra
% Email: kapil@nitkkr.ac.in/navkapil@gmail.com/navprapil@gmail.com

%% Variable Information
% file: threecluster_lineeq.mat keeps dataset variable
% dataset: rectangular matrix with last column to be class label, and each
% row repersents a sample
% redKer flag to switch on (1) /off (0) reduced kernel usage


close all
clear
file = "threecluster_lineeq"; 
load(strcat(file,".mat"));
createPlot = 1;
samples = size(dataset,1);
dataset = dataset(randperm(samples),:);
redKer = 1; % reduced kernel flag, if set use reduced kernel

%% Train-Test data separation 
% initial rows will form train data and later will form test
TrTr = 0.9; %train to test ratio 
train_num = round(TrTr*samples);
test_num = samples-train_num;
train_set = dataset(1:train_num,:);
test_set = dataset(train_num+1:end,:);
%% parameter setting for reduced kernel
if redKer == 1
    reduction = 0.01;       % fraction of samples to be considered in reduced kernel
    redSize = round(train_num*reduction);
    redKerIn = randperm(train_num,redSize);
else
    redKerIn = 1:train_num;
end
%% (un)labelled samples separation for semi supervised version
% initial rows form labelled and later rows form unlabelled data
lbld_unlbld_ratio = 0.005;  
lbld_train_samples = floor(lbld_unlbld_ratio*train_num);
lbld_train_set = train_set(1:lbld_train_samples,:);
unlbld_train_set = train_set(lbld_train_samples+1:end,:);

%% collect all the data that undergo division during cross validation
foldabledata = seprate_class(lbld_train_set);

% [c1,c3,c5,delta] = para{:};
c1vs = [1005];%10.^(-8:10:8);
c3vs = [105];%10.^(-8:10:8);
c5vs = [100];%[1005];%10.^(-8:10:8);%pass only 0 value for LSKSVCR 
deltavs = [0.5];%(0.1:0.2:0.9);
knlvs = [0.3];%[0.01,0.05,0.1,0.2];

%% parameterlist keeps all various values of parameters that are tuned 
% these values are grid searched for best effectiveness of the model
parameterlist(1).values = c1vs;
parameterlist(2).values = c3vs;
parameterlist(3).values = c5vs;
parameterlist(4).values = deltavs;
parameterlist(5).values = knlvs;

%% parameters that will stay constant throughout the crossvalidation
% it might keep the parameters that are not considered for cross validation
% and it keep the model training function(@train_func) and prediction
% function (@test_func) and evaluator of performance parameters (@prfrmncePara).
% k number of fold considered for k-fold cross validation.
otherpara.graph_k = 3;
otherpara.graph_mu = 0.02;
otherpara.U = unlbld_train_set;
otherpara.knl = @rbf_kernel;%@linear_kernel; %


train_func = @fitNonLinLapLSKSVCR;
test_func = @predictNonLinLapLSKSVCR;
prfrmncePara = @confusionmatStats;

fold = 2;

[optpara] = crossvalidation(foldabledata,otherpara,parameterlist, ...
    train_func,test_func, prfrmncePara,fold);
save(strcat(file,"_non_lin_res.mat"),"optpara",'-mat')

if redKer==1
    otherpara.redKerIn = redKerIn;% reduced kernel matrix only used for final training and testing
end
%% retrain on the whole training data and generate the model
tic
[Mdl] = train_func(foldabledata,otherpara,optpara.para);
train_time = toc;        

%% get the final performance on test data
% testfold.data = test_set;
tic
[res] = test_func(foldabledata,test_set,optpara.para,Mdl,otherpara);
test_time = toc;

[finalPerformance] = prfrmncePara(res.act,res.PC);
finalPerformance.train_time = train_time;
finalPerformance.test_time = test_time;
save(strcat(file,"_non_lin_res.mat"),"optpara","finalPerformance",'-mat')

%% create plot if asked

if (createPlot)
    hold on
    NonLinLbldNlbld_plot_decision_bndry(foldabledata,otherpara, optpara,Mdl,test_func)
    if optpara.para(3)~=0 %plot unlabelled data only in case of lap-ls k-scvr
        plot(otherpara.U(:,1),otherpara.U(:,2),'k.') 
    end
end

