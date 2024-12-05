function [res] = predict_NonLinLapKSVCR(foldableTrainData,testfold,parameter,Mdl,otherpara)
    A = foldableTrainData(3).data(:,1:end-1);
    B = foldableTrainData(1).data(:,1:end-1);
    C = foldableTrainData(2).data(:,1:end-1);

    k = otherpara.k; % Graph laplacian parameter as k-nn
    U = otherpara.U(:,1:end-1);
    
    u = Mdl.u;
    knl = otherpara.knl;

    M = [A;B;C;U];

    test_set = [testfold(1).data;testfold(2).data;testfold(3).data];
    indx = randperm(size(test_set,1));
    test_set = test_set(indx,:);
    
    para = num2cell(parameter);
    [c1,c2,c3,delta,knl_para] = para{:};
    % c2 = c1; 
    mu=knl_para;

    test_f = [knl(test_set(:,1:end-1),M,mu) ones(size(test_set,1),1)];
    % k_testf = knl(test_f,M,mu);
    d = test_f*u;
    
    I3 = d > (1+delta)/2;
    I1 = d < -(1+delta)/2;
    % IZ = ones(size(test,1),1)-or(IP,IN);
    res.PC = I3-I1;
    res.act = test_set(:,end);
end