function[optpara] = crossvalidation(foldabledata,otherpara,parameterlist,train_func,test_func, prfrmncePara,fold)
%--------------------------------------------------------------------------
%  foldabledata: A structure array, at each index a 2-dim array of input 
%                samples having features and target value with attribute
%                .data
%   example: foldabledata(1).data=[rand(100,4) 2*ones(100,1)]
%  
%  fold: parameter k of k-fold crossvalidation
%  
%  otherpara: structure of all other parameters needs to be passed to
%               trainfunc or testfunc functions, these will be passed
%               exactly in same way
%  
%  parameterlist: structure of parameters on which crossvalidaion to be
%               performed (parameter values are kept as in below example)
%       parameterlist(1).values = [1,2,3];
%       parameterlist(2).values = [4,7,8,3];
%  
%  train_func : handle to train function returns out structure which will be
%               feeded to testfunc
%
%  test_func  : handle to predict function, returns structure containing an 
%               PC field, considered for predicted value and act field for
%               actual target values
%
%  prfrmncePara: handle to preformance parameter evaluation function on 
%                the lower value of which hyperparameter is tuned. 
%                It should output a structure with attribute err
%--------------------------------------------------------------------------
%Output:
%   optpara.parameter- structure that keeps parameter values asked to tuned for
%   optpara.error-structure that keeps values of error corresponding to each
%                  specific set of parameter values asked to tuned for 
% Author: Kapil Email: Kapil@nitkkr.ac.in
%--------------------------------------------------------------------------
minerr = Inf;
nloop = 1;
for i = 1:size(parameterlist,2)
    nloop = nloop*size(parameterlist(i).values,2);
end
index = zeros(1,size(parameterlist,2));
parameter = zeros(nloop,size(parameterlist,2));

% This loop is creating all possible combinations of hyperparameters and putting it in the rows of 
% parameter
for i = 0:nloop-1
    temp = i;
    for j = size(parameterlist,2):-1:1
        index(j) = mod(temp,size(parameterlist(j).values,2))+1;
        temp = floor(temp/size(parameterlist(j).values,2));
        parameter(i+1,j) = parameterlist(j).values(index(j));
    end
end
error = zeros(nloop,1);
parfor i = 0:nloop-1 %parfor may be replaced by for, if you don't want parallelism
    parameter(i+1,:)
    for j = 1:fold
        % creates fold number of division of entire data in foldabledata and the j-th division in testfold and all other in trainfold
        [trainfold,testfold] = blocking(foldabledata,fold,j); 

        % train the model on trainfold
        [Mdl] = train_func(trainfold,otherpara,parameter(i+1,:));
        
        % get the prediction on testfold
        [res] = test_func(trainfold,testfold,parameter(i+1,:),Mdl,otherpara);

        % get the values of performance parameters
        [val] = prfrmncePara(res.act,res.PC);

        % accumulate the error for all iterations of crossvalidation
        error(i+1) = error(i+1) + val.err;
    end
end
[~,indx] = min(error);
optpara.para = parameter(indx,:);
optpara.minerr = error(indx);
optpara.parameter = parameter; % all parameter combinations
optpara.error = error; % error for each combination of parameter values
%optpara.Mdl = Mdl;
