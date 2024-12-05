function[c] = seprate_class(dataset)
%Separates the dataset on the basis of last column.
%   last column assumed to have numeric labels fixed and only few.
%   input: dataset for classification
%   output: array of structure, each cell having one classes' info with
%     .data: keeps the features of all the samples of a particular class
%     along with label
%     .label: keeps the original label
%     .count: keeps the number of samples in the class
%     .ori_index: keeps the original index (in dataset/input) of each
%                 sample in (.data) in the order
%   Example call: c = seprate_class(dataset)
%
%   Author: Kapil(kapil@nitkkr.ac.in)

    [~,n] = size(dataset);
    lab = unique(dataset(:,n));
    num_class = length(lab);
    c(num_class) = struct('label',[],'data',[],'count',[],'ori_index',[]);
    for i=1:num_class
        a = (dataset(:,n)==lab(i));
        c(i).data = dataset(a,:);
        c(i).label = lab(i);
        c(i).count = sum(a);
        c(i).ori_index = find(a);
    end
end