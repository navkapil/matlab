function [confusionMatrix, precision,recall,fscore] = calPerPara(trueLabels, predictedLabels)

    % Get unique class labels
    classes = unique(trueLabels);
    numClasses = numel(classes);

    % Initialize confusion matrix
    confusionMatrix = zeros(numClasses);

    % Populate confusion matrix
    for i = 1:numClasses
        trueClass = classes(i);
        trueIndices = trueLabels == trueClass;

        for j = 1:numClasses
            predictedClass = classes(j);
            predictedIndices = predictedLabels == predictedClass;

            % Count number of occurrences where true label and predicted label match
            confusionMatrix(i, j) = sum(trueIndices & predictedIndices);
        end
    end

    % Calculate precision, recall, and F-score for each class
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    fscore = zeros(numClasses, 1);

    for i = 1:numClasses
        truePositive = confusionMatrix(i, i);
        falsePositive = sum(confusionMatrix(:, i)) - truePositive;
        falseNegative = sum(confusionMatrix(i, :)) - truePositive;

        % Calculate precision, recall, and F-score
        precision(i) = truePositive / (truePositive + falsePositive);
        recall(i) = truePositive / (truePositive + falseNegative);
        fscore(i) = 2*truePositive/(2*truePositive + falsePositive + falseNegative);
    end
end