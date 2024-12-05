function [trainfold,testfold] = blocking(foldabledata,fold,i)
%--------------------------------------------------------------------------
% Input:
%   data = this should be divided in 'fold' blocks and ith block should be 
%       test and other should be taken as train starting from 1 to value of 
%       'fold'
%--------------------------------------------------------------------------
%Output:
%   trainfold = train set for ith iteration, structure of foldable data
%   maintained (stratified sampling for each fold)
%   testfold = test set for ith iteration
%--------------------------------------------------------------------------
    if i>fold
        disp(strcat('Block ',i,' has excceded folds = ',fold));
        exit;
    end
    testfold = struct([]);
    trainfold = struct([]);

    numclass = length(foldabledata);
    for j=1:numclass
        data = foldabledata(j).data; % the data structure as managed by seprate_class
        samples = size(data,1);
        % ori_index = foldabledata(j).ori_index;
        % count = foldabledata(j).count;
        % label = foldabledata(j).label;

        blocksz = samples/fold;
        teststart = floor(blocksz*(i-1)+1);
        testend = floor(blocksz*(i));
        
        testfold(j).data = data(teststart:testend,:);
        
        % testfold(j).label = foldabledata(j).label;
        % testfold(j).count = testend-teststart+1;
        % testfold(j).ori_index = ori_index(teststart:testend,:);

        trainfold(j).data = [data(1:teststart-1,:);data(testend+1:samples,:)];
        
        % trainfold(j).label = label;
        % trainfold(j).count = count-testfold(j).count;
        % trainfold(j).ori_index = [ori_index(1:teststart-1,:);ori_index(testend+1:samples,:)];
    end 
end