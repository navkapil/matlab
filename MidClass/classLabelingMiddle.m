function[c] = classLabelingMiddle(c)
%Separates the dataset on the basis of last column.
%   input: separated classes, output of c=seprate_class(dataset) with
%     .data: keeps the features of all the samples of a particular class
%     .label: keeps the original label
%     .count: keeps the number of samples in the class
%     .ori_index: keeps the original index (in dataset/input) of each
%                 sample in (.data) in the order
%     .centroid: tells the centroid of each class cloud
%
%   output: 
%     c.midl: keeps proposed label assignment with mean method  
%     midl: keeps label assignment to use it in KSVCR
%   
%   Author: Kapil(kapil@nitkkr.ac.in)

    m_c12 = (c(1).centroid+c(2).centroid)/2;
    m_c13 = (c(1).centroid+c(3).centroid)/2;
    m_c23 = (c(2).centroid+c(3).centroid)/2;
    [~,index] = min([norm(m_c23-c(1).centroid), norm(m_c13-c(2).centroid), norm(m_c12-c(3).centroid)]);
    
    c(index).midl = zeros(1,1);
    c(mod(index-1+1,3)+1).midl = 1;
    c(mod(index-1-1,3)+1).midl = -1;
end
