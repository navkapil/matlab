function [neutral] = cosine_method(train_data,knl,knl_para)
% train_data(i).data has a matrix with features in columns and last column
% to be class label
% knl: handle to kernel function
% knl_para: kernel parameter
% ----------------------------------------
% It returns the class whose centroid if taken as vertex of the angle, then 
% show the least value of the dot product (higher negative value shows that 
% the two classes are in the opposite direction of the pivotal class) 

    num_classes = length(train_data);
    cd = struct([]);
    for i=1:num_classes
        cd(i).n = size(train_data(i).data,1);
        cd(i).e = ones(cd(i).n,1);
        cd(i).f = train_data(i).data(:,1:end-1);
    end
    dist = zeros(num_classes,1);
    for i = 1:num_classes
        A = i;
        B = mod(i,3)+1;
        C = mod(i+1,3)+1;
        h1 = [-cd(A).e'/cd(A).n  cd(B).e'/cd(B).n 0*cd(C).e'];
        h2 = [-cd(A).e'/cd(A).n  0*cd(B).e' cd(C).e'/cd(C).n];
        D = [cd(A).f;cd(B).f;cd(C).f];
        K = knl(D,D,knl_para);
        dist(A) = h1*K*h2';
    end
    [~,indx] = min(dist);
    neutral.index = indx; % index of the class that should come out to be neutral
    neutral.label = train_data(j).data(1,end); % label of the class that should come out to be neutral
end
