function [neutral] = mid_point(train_data,knl,knl_para)
    % train_data(i).data has a matrix with features in columns and last column
    % to be class label
    % knl: handle to kernel function
    % knl_para: kernel parameter
    % ----------------------------------------
    % It returns the class label which is nearest to the mid way between the
    % other two class centroids
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
        h = [-cd(A).e'/cd(A).n  cd(B).e'/(2*cd(B).n) cd(C).e'/(2*cd(C).n)]; 
        D = [cd(A).f;cd(B).f;cd(C).f];
        K = knl(D,D,knl_para);
        dist(A) = h*K*h';
    end
    [~,neutral] = min(dist);
    % neutral = train_data(j).data(1,end);
end