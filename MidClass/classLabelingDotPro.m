function[dotp,c] = classLabelingDotPro(c)
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
%     c.dotp: keeps proposed label assignment with dot product method  
%     dotp: keeps label assignment to use it in KSVCR
%   
%   Author: Kapil(kapil@nitkkr.ac.in)
    
    num_class = length(c);
    dim = size(c(1).traindata,2);
    d = zeros(dim,num_class,num_class);
    dotp=zeros(num_class,1);
    for tail=1:num_class
        for head=1:num_class
            d(:,tail,head) = c(head).centroid-c(tail).centroid;
        end 
        dotp(tail) = d(:,tail,mod(tail-1+1,3)+1)'* d(:,tail,mod(tail-1-1,3)+1);
    end
    
    [~,index] = min(dotp);
    
    c(index).dotp = zeros(1,1);
    c(mod(index-1+1,3)+1).dotp = 1;
    c(mod(index-1-1,3)+1).dotp = -1;
end
