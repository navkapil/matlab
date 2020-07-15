function demoSVM(dataset)
% [m,n] = size(dataset);
trainX = dataset(:,1:end-1);
trainY = dataset(:,end);
classes = unique(trainY);

if(length(classes)~=2)
    disp('not a binary class problem');
    return;
end
trainY = -(trainY==classes(1))+(trainY==classes(2));
knl=@rbf_kernel;
C=10;
knl_para = .0125;

[alpha, b] = fit_kernel_SVM(trainX,trainY,knl,knl_para,C);

plot_decision_bndry(alpha,b,trainX,trainY,knl,knl_para);
end