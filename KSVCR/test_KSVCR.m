function [acc,PC,score,orientation] = test_KSVCR(model,testdata, testlabel)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    testsample = size(testdata,1);
    T = testdata;
    %kernel
    if strcmp(model.kernel,'linear')
        %linear
        KT = T*model.P';
    elseif strcmp(model.kernel,'rbf') 
        %non-linear
        ker = @rbf_kernel;
        KT = ker(T,model.P,model.mu);
    end 
    orientation = model.P'*model.J'*model.gamma; 
    score = KT*model.J'*model.gamma+model.b;
    
    PC = (score>(model.delta+1)/2)-(score<-(model.delta+1)/2);
    if size(testlabel,1) == 0
        acc=-1;
    else
        acc = sum(PC==testlabel)/testsample;
    end
end