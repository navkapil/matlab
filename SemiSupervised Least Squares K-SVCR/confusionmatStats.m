function stats = confusionmatStats(group,grouphat)
% INPUT
% group = true class labels
% grouphat = predicted class labels
%
% OR INPUT
% stats = confusionmatStats(group);
% group = confusion matrix from matlab function (confusionmat)
%
% OUTPUT
% stats is a structure array
% stats.confusionMat
%               Predicted Classes
%                     p'      n'
%              ___|_______|_______| 
%       Actual  p |  TP   |  FN   |
%      Classes  n |  FP   |  TN   |
%
% stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
% stats.precision = TP / (TP + FP)                  % for each class label
% stats.sensitivity = TP / (TP + FN)                % for each class label
% stats.specificity = TN / (FP + TN)                % for each class label
% stats.recall = sensitivity                        % for each class label
% stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
%
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% 
field1 = 'confusionMat';
if nargin < 2
    value1 = group;
else
    [value1,gorder] = confusionmat(group,grouphat,'Order',[-1 0 1]);
    % group: group labels-- actual labels
    % grouphat: predicted labels
end
numOfClasses = size(value1,1);
totalSamples = sum(sum(value1));

%initalise all the variables to zero at once
[TP,TN,FP,FN,accuracy,sensitivity,specificity,precision,f_score,numSamples] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
   TP(class) = value1(class,class);
   tempMat = value1;
   tempMat(:,class) = []; % remove column
   tempMat(class,:) = []; % remove row
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(value1(:,class))-TP(class);
   FN(class) = sum(value1(class,:))-TP(class);
end
for class = 1:numOfClasses
    accuracy(class) = (TP(class) + TN(class)) / totalSamples;
    sensitivity(class) = TP(class) / (TP(class) + FN(class));
    specificity(class) = TN(class) / (FP(class) + TN(class));
    precision(class) = TP(class) / (TP(class) + FP(class));
    f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
    numSamples(class)=sum(value1(class,:));
end
field2 = 'accuracy';  value2 = accuracy;
field3 = 'sensitivity';  value3 = sensitivity;
field4 = 'specificity';  value4 = specificity;
field5 = 'precision';  value5 = precision;
field6 = 'recall';  value6 = sensitivity;
field7 = 'Fscore';  value7 = f_score;
field8 = 'Mean_acc'; value8 = mean(accuracy);
field9 = 'Mean_precision'; value9 = mean(precision);
field10 = 'Mean_recall';value10=mean(sensitivity);
field11 = 'Mean_Fscore';value11=mean(f_score);
field12='Wtd_acc';value12=numSamples'*accuracy/totalSamples;
field13 = 'Wtd_precision'; value13=numSamples'*precision/totalSamples;
field14 = 'Wtd_recall';value14=numSamples'*sensitivity/totalSamples;
field15 = 'Wtd_Fscore'; value15 = numSamples'*f_score/totalSamples;
field16 = 'err';value16 = -value11;
stats = struct(field1,value1,field2,value2,field3,value3,field4,value4, ...
    field5,value5,field6,value6,field7,value7,field8,value8,field9,value9, ...
    field10,value10,field11, value11,field12,value12,field13,value13, ...
    field14,value14,field15,value15, field16, value16);
if exist('gorder','var')
    stats = struct(field1,value1,field2,value2,field3,value3,field4,value4, ...
        field5,value5,field6,value6,field7,value7,field8,value8,field9,value9, ...
    field10,value10,field11, value11,field12,value12,field13,value13, ...
    field14,value14,field15,value15, field16, value16,'groupOrder',gorder);
end
end